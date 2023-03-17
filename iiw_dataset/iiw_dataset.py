import torch
from PIL import Image
from os.path import exists, join, split
import numpy as np
import json
import utils.data_transforms_iiw as transforms

class IIWDataset(torch.utils.data.Dataset):
    def __init__(self,transforms=None, data_dir = 'iiw_dataset/',
                 split = 'train',out_name = True, guidance = True, guide_size=128):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_list = None
        self.image_data_list = data_dir + '/data'
        self.split = split
        self.read_lists()
        self.out_name = out_name
        self.guidance = guidance
        self.guide_size = guide_size
        
        
    def get_label(self,label_path):
        
        #darker_dict = {'E':0,'C':1,'W':2}
        label_data = json.load(open(label_path,'rb'))
        comparisons = label_data['intrinsic_comparisons']
        points = label_data['intrinsic_points']
        point_dict = {}
        for point in points:
            point_dict[point['id']] = [point['x'],point['y']]
        comparison_point = [] 
        comparison_label = []  
        for pair in comparisons:
            comparison_point.append([point_dict[pair['point1']],
                                    point_dict[pair['point2']]])
            comparison_label.append([pair['darker'],pair['darker_score']])
        return comparison_point, comparison_label      
    

    def __getitem__(self, index):
        img_name = self.image_list[index] + '.png'
        label_name = self.image_list[index] + '.json'
        data = [Image.open(join(self.image_data_list, img_name))]
        data = np.array(data[0])
        if len(data.shape) == 2: # gray scale
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        #get label
        label_dir = join(self.image_data_list, label_name)
        points, labels = self.get_label(label_dir)
        if self.guidance:
            # from IPython import embed
            # embed()
            guide_image = data[0].resize((self.guide_size, self.guide_size),Image.ANTIALIAS)
            guide_image = self.transforms.transforms[-2](guide_image)[0]
            guide_image = self.transforms.transforms[-1](guide_image)[0]
            
        data = list(self.transforms(data[0],np.array(points)))
        data.append(labels)

        if self.out_name:
            data.append(self.image_list[index])

        if self.guidance:
            data.append(guide_image)
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.data_dir, self.split+'.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]



if __name__ == '__main__':
    
    info = json.load(open(join('iiw_dataset/', 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    crop_size = 512
    naive_t = [transforms.Resize(crop_size),
            transforms.ToTensorMultiHead(),
            normalize]
    
    trainset = IIWDataset(transforms=transforms.Compose(naive_t))
    max_point_num = 0
    all_data = trainset.__getitem__(0)


