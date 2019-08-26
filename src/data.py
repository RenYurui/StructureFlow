import torch.utils.data as data
import torch
import os
import os.path
import glob
from torchvision import transforms
import torchvision.transforms.functional as transFunc
import random
import numpy as np
import torch.nn.functional as F
import math
import scipy.io as scio
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
import cv2


## TODO: choose with or without transformation at test mode
class Dataset(data.Dataset):
    def __init__(self, gt_file, structure_file, config, mask_file=None):
        self.gt_image_files = self.load_file_list(gt_file)
        self.structure_image_files = self.load_file_list(structure_file)

        if len(self.gt_image_files) == 0:
            raise(RuntimeError("Found 0 images in the input files " + "\n"))

        if config.MODE == 'test':
            self.transform_opt = {'crop': False, 'flip': False,
                            'resize': config.DATA_TEST_SIZE, 'random_load_mask': False}
            config.DATA_MASK_TYPE == 'from_file' if mask_file is not None else config.DATA_MASK_TYPE 
        else:
            self.transform_opt = {'crop': config.DATA_CROP, 'flip': config.DATA_FLIP,
                            'resize': config.DATA_TRAIN_SIZE, 'random_load_mask': True}
                            
        self.mask_type = config.DATA_MASK_TYPE
        # generate random rectangle mask 
        if self.mask_type == 'random_bbox':
            self.mask_setting = config.DATA_RANDOM_BBOX_SETTING
        # generate random free form mask 
        elif self.mask_type == 'random_free_form':
            self.mask_setting = config.DATA_RANDOM_FF_SETTING
        # read masks from files
        elif self.mask_type == 'from_file':
            self.mask_image_files = self.load_file_list(mask_file)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.gt_image_files[index])
            item = self.load_item(0)
        return item

    def __len__(self):
        return len(self.gt_image_files)  


    def load_file_list(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []        


    def load_item(self, index):
        gt_path = self.gt_image_files[index]
        structure_path = self.structure_image_files[index]
        gt_image = loader(gt_path)
        structure_image = loader(structure_path)
        transform_param = get_params(gt_image.size, self.transform_opt)
        gt_image, structure_image = transform_image(transform_param, gt_image, structure_image)

        inpaint_map = self.load_mask(index, gt_image)
        input_image = gt_image*(1-inpaint_map)

        return input_image, structure_image, gt_image, inpaint_map


    def load_mask(self, index, img):
        _, w, h = img.shape
        image_shape = [w, h]
        if self.mask_type == 'random_bbox':
            bboxs = []
            for i in range(self.mask_setting['num']):
                bbox = random_bbox(self.mask_setting, image_shape)
                bboxs.append(bbox)
            mask = bbox2mask(bboxs, image_shape, self.mask_setting)
            return  torch.from_numpy(mask)

        elif self.mask_type == 'random_free_form':
            mask = random_ff_mask(self.mask_setting, image_shape)
            return torch.from_numpy(mask)

        elif self.mask_type == 'from_file':
            if self.transform_opt['random_load_mask']:
                index = np.random.randint(0, len(self.mask_image_files))
                mask = gray_loader(self.mask_image_files[index])
                if random.random() > 0.5:
                    mask = transFunc.hflip(mask)
                if random.random() > 0.5:
                    mask = transFunc.vflip(mask)  
            else:
                mask = gray_loader(self.mask_image_files[index])
            mask = transFunc.resize(mask, size=image_shape)  
            mask = transFunc.to_tensor(mask)      
            return mask
        else:
            raise(RuntimeError("No such mask type: %s"%self.mask_type))

    def load_name(self, index, add_mask_name=False):
        name = self.gt_image_files[index]
        name = os.path.basename(name) 

        if not add_mask_name:
            return name
        else:
            if len(self.mask_image_files)==0:
                return name
            else:
                mask_name = os.path.basename(self.mask_image_files[index])
                mask_name, _ = os.path.splitext(mask_name) 
                name, ext = os.path.splitext(name)                               
                name = name+'_'+mask_name+ext
                return name 



    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item 
                

def random_bbox(config, shape):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including DATA_NEW_SHAPE,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    img_height = shape[0]
    img_width = shape[1]
    height, width = config['shape']
    ver_margin, hor_margin = config['margin']
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    t = np.random.randint(low=ver_margin, high=maxt)
    l = np.random.randint(low=hor_margin, high=maxl)
    h = height
    w = width
    return (t, l, h, w)

def random_ff_mask( config, shape):
    """Generate a random free form mask with configuration.

    Args:
        config: Config should have configuration including DATA_NEW_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)
    """

    h,w = shape
    mask = np.zeros((h,w))
    num_v = 12+np.random.randint(config['mv']) #tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1+np.random.randint(5)):
            angle = 0.01+np.random.randint(config['ma'])
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10+np.random.randint(config['ml'])
            brush_w = 10+np.random.randint(config['mbw'])
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    return mask.reshape((1,)+mask.shape).astype(np.float32)


def bbox2mask( bboxs, shape, config):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including DATA_NEW_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    height, width = shape
    mask = np.zeros(( height, width), np.float32)
    #print(mask.shape)
    for bbox in bboxs:
        if config['random_size']:
            h = int(0.1*bbox[2])+np.random.randint(int(bbox[2]*0.2+1))
            w = int(0.1*bbox[3])+np.random.randint(int(bbox[3]*0.2)+1)
        else:
            h=0
            w=0
        mask[bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
    #print("after", mask.shape)
    return mask.reshape((1,)+mask.shape).astype(np.float32) 
      

def gray_loader( path):
    return Image.open(path)

def loader( path):
    return Image.open(path).convert('RGB')

def get_params(size, transform_opt):
    w, h = size
    if transform_opt['flip']:
        flip = random.random() > 0.5
    else:
        flip = False
    if transform_opt['crop']:
        transform_crop = transform_opt['crop'] \
        if w>=transform_opt['crop'][0] and h>=transform_opt['crop'][1] else [h, w]
        x = random.randint(0, np.maximum(0, w - transform_crop[0]))
        y = random.randint(0, np.maximum(0, h - transform_crop[1]))
        crop = [x, y, transform_crop[0], transform_crop[1]]
    else:
        crop = False
    if transform_opt['resize']:
        resize = [transform_opt['resize'], transform_opt['resize'],]
    else:
        resize = False
    param = {'crop': crop, 'flip': flip, 'resize': resize}
    return param

def transform_image(transform_param, gt_image, structure_image, normalize=True, toTensor=True):
    transform_list = []

    if transform_param['crop']:
        crop_position = transform_param['crop'][:2]  
        crop_size = transform_param['crop'][2:]  
        transform_list.append(transforms.Lambda(lambda img: __crop(img, crop_position, crop_size)))
    if transform_param['resize']:
        transform_list.append(transforms.Resize(transform_param['resize']))
    if transform_param['flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, True)))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    trans = transforms.Compose(transform_list)  
    if gt_image.size != structure_image.size:
        structure_image = transFunc.resize(structure_image, size=gt_image.size)

    gt_image = trans(gt_image)
    structure_image = trans(structure_image)

    return gt_image, structure_image  

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


