from cmath import nan
import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from albumentations import Compose
from natsort import natsorted
from datasets.preprocess import augment, normalization
from datasets.stereo_albumentation import RGBShiftStereo, RandomBrightnessContrastStereo, random_crop, random_crop_fixed


class Deep360Dataset(data.Dataset):

    def __init__(self, datadir, split='train'):
        super(Deep360Dataset, self).__init__()
        self.datadir = datadir
        self.split = split
        if split == 'train':
            self.sub_folder = 'training'
        elif split == 'validation':
            self.sub_folder = 'validation'
        elif split == 'test':
            self.sub_folder = 'testing'
        # all frames
        self.frames_dir = os.listdir(self.datadir)
        self.data_list = []
        self._read_data()
        self._augmentation()

    def _read_data(self):
        assert self.frames_dir is not None
        for frame_dir in self.frames_dir:
            rgb_path = os.path.join(
                self.datadir, frame_dir, self.sub_folder, 'rgb')
            disp_path = os.path.join(
                self.datadir, frame_dir, self.sub_folder, 'disp')
            files = os.listdir(disp_path)
            names = [file[0:-9] for file in files]
            for name in names:
                item_data = {
                    'left_rgb_path': os.path.join(rgb_path, f'{name}_rgb{name[-2]}.png'),
                    'right_rgb_path': os.path.join(rgb_path, f'{name}_rgb{name[-1]}.png'),
                    'disp_path': os.path.join(disp_path, f'{name}_disp.npz'),
                }
                self.data_list.append(item_data)

    def _augmentation(self):
        if self.split == 'train':
            self.transformation = Compose([
                RGBShiftStereo(always_apply=True, p_asym=0.5),
                RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
            ])
        elif self.split == 'validation' or self.split == 'test' or self.split == 'validation_all':
            self.transformation = None
        else:
            raise Exception("Split not recognized")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_data = {}
        data_dict = self.data_list[idx]

        # left
        left_fname = data_dict['left_rgb_path']
        left = np.array(Image.open(left_fname).convert('RGB')).astype(np.uint8)
        input_data['left'] = left

        # right
        right_fname = data_dict['right_rgb_path']
        right = np.array(Image.open(
            right_fname).convert('RGB')).astype(np.uint8)
        input_data['right'] = right

        # disp
        disp_fname = data_dict['disp_path']
        disp = np.load(disp_fname)['arr_0']

        input_data['disp'] = disp
        input_data['occ_mask'] = np.zeros_like(disp).astype(np.bool)


        if not self.split == 'test':  # no disp for test files
            input_data = augment(input_data, self.transformation)
        else:
            input_data = augment(input_data, None)

        return input_data
