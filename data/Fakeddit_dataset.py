from resource_paths import resolve_read_path as _read_path
import cv2

import torch
import torch.utils.data as data

from .util import read_img, channel_convert

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Fakeddit_dataset(data.Dataset):

    def __init__(self, root_path='../dataset', mode='train', image_size=224):
        super().__init__()
        root_path = os.fspath(root_path)
        self.label_dict = []
        self.image_size = image_size
        self.image_path = root_path+'/Fakeddit/images/images'
        self.image_path = os.path.abspath(_read_path(self.image_path))
        list_dirs = os.listdir(self.image_path)
        self.image_listdir = {}
        for list_dir in list_dirs:
            idx = list_dir.rfind('.')
            self.image_listdir[list_dir[:idx]] = list_dir

        train_path = root_path + '/Fakeddit/multimodal_' + mode +"_cleaned"+'.tsv'
        df = pd.read_csv(_read_path(train_path), sep="\t",
                         dtype={'id': str, 'clean_title': str})
        df = df.replace(np.nan, '', regex=True)
        df.fillna('', inplace=True)
        
        positive_records, negative_records = [], []
        rows = len(df.index)
        for i in tqdm(range(0, rows)):
            if df['hasImage'][i] == False:
                continue
            try:
                
                if df['id'][i] in self.image_listdir:
                    images_name = df['id'][i] + ".jpg"
                else: continue
            except KeyError:
                continue
            # Binary labels: real = 0, fake = 1.
            label = 0 if (df['2_way_label'][i]==1) and (df['3_way_label'][i]==0) and (df['6_way_label'][i]==0) else 1
            content = df['clean_title'][i]
            record = {}
            record['images'] = images_name
            record['label'] = label
            record['content'] = content
            if label == 0:
                negative_records.append(record)
            else:
                positive_records.append(record)
        self.label_dict = positive_records + negative_records

        assert len(self.label_dict)!=0, 'Error: GT path is empty.'

    def __getitem__(self, index):

        GT_size = self.image_size

        # Select an image for this record.
        record = self.label_dict[index]
        images, label, content = record['images'], record['label'], record['content']

        imgs = images.split(',')
        GT_path = imgs[np.random.randint(0, len(imgs))]

        try:
            if GT_path in self.image_listdir:
                GT_path = self.image_listdir[GT_path]

            GT_path = "{}/{}".format(self.image_path,GT_path)
            img_GT = read_img(GT_path)
        except Exception:
            raise IOError("Load {} Error {}".format(GT_path, record['images']))

        img_GT = channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]

        # Resize without cropping.
        img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                            interpolation=cv2.INTER_LINEAR)

        # Reverse the channel order and convert HWC to a CHW tensor.
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        return content, img_GT, label, GT_path

    def __len__(self):
        return len(self.label_dict)
