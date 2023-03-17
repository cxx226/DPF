import torch
from torch import nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.vit import forward_flex
from model.edsr import make_edsr_baseline
from model.blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    ranged in [-1, 1]
    e.g. 
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [H*W, 2]
    return ret


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=128,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = path_1

        for i, it in enumerate(self.scratch.output_conv):
            out = it(out)
            if i == 3:
                lr_feature = out.clone()
            if i == 4:
                lr_map = out.clone()

        """ 
        NOTE:
        image-x, depth - lr_map, coord(not yet), res - lr_map.squeeze(),
        """
        
        return [out], [lr_map],[lr_feature]


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 128

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class IIF(nn.Module):
    # modified from: https://github.com/ashawkey/jiif/blob/main/models/jiif.py
    def __init__(self, num_classes, feat_dim=128, guide_dim=128,mlp_dim=[512,256,128,64]): 
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.classes = num_classes
        imnet_in_dim = self.feat_dim + self.guide_dim * 2 + 2
        self.image_encoder = make_edsr_baseline(n_feats=self.guide_dim, n_colors=3)
        dim = num_classes + 1
        self.imnet = MLP(imnet_in_dim, out_dim=dim, hidden_list=self.mlp_dim) # out_dim=num_class
        
    def query(self, feat, coord, hr_guide, lr_guide, num_classes):
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W
        b, c, h, w = feat.shape # lr
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).contiguous().unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous() # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []
        
        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx 
                coord_[:, :, 1] += (vy) * ry 
                k += 1
                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous() # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous() # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous() # [B, N, C]
                q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)

                inp = torch.cat([q_feat, q_guide, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1) # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1) # [B, N, 2, kk]

        weight = F.softmax(preds[:,:,num_classes,:], dim=-1)

        ret = torch.zeros_like(preds[:,:,0:num_classes,:])

        for i in range(num_classes):
            ret[:,:,i,:] = (preds[:,:,i,:] * weight)
        
        Ret = ret.sum(-1, keepdim=True).squeeze(-1)
        
        return Ret

    def forward(self, image, lr_res, res, lr_image_feature, lr_image=None, batched=True):
        bs = image.shape[0]
        
        scale = image.shape[2]/res.shape[2]
        res = F.interpolate(res,scale_factor=scale)
        res = res.view(bs,lr_res.shape[1],-1,1)
        res = res.squeeze(-1)
        
        coord = make_coord(image.shape[-2:],flatten=True).unsqueeze(0).repeat(bs,1,1).cuda()
        
        hr_guide = self.image_encoder(image)
        if lr_image is None:
            lr_guide = lr_image_feature
        else:
            lr_guide = self.image_encoder(lr_image)


        feat = lr_image_feature 

        if not batched:
            res = res + self.query(feat, coord, hr_guide, lr_guide)
        # batched evaluation to avoid OOM
        else:
            N = coord.shape[1] # coord ~ [B, N, 2]
            n = 30720
            tmp = []
            for start in range(0, N, n):
                end = min(N, start + n)
                ans = self.query(feat, coord[:, start:end], hr_guide, lr_guide, self.classes) # [B, N, 60]
                tmp.append(ans)
            res = res + torch.cat(tmp, dim=1).permute(0,2,1).contiguous()

        img_size = image.shape[-2:]
        res = res.reshape(bs,self.classes,img_size[0],img_size[1])
        
        return res

class DPF(torch.nn.Module):
    def __init__(self, num_classes, guide = True, backbone = None,**kwargs):
        super(DPF, self).__init__()
        # Initialize dense prediction backbone
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = DPTSegmentationModel(num_classes, backbone="vitb_rn50_384")
        self.guide = guide
        if self.guide:
            self.guide_model =  IIF(num_classes)

    def forward(self, x, input_guide=None,continous=True):
        output,lr_map, lr_feature = self.backbone(x)

        if continous: # predict continuous value, used for IIW 
            for idx in range(len(output)):
                output[idx] = output[idx].sigmoid()

        guide_output = None
        if self.guide:
            guide_output = self.guide_model(input_guide,lr_map[0],output[0],lr_feature[0])
        return output[0], guide_output