
from resource_paths import resolve_read_path as _fnd_read_path
import cv2

import torch
import torch.utils.data as data

from .util import read_img, channel_convert
import random
import torchvision.transforms.functional as F

from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Fakeddit_dataset(data.Dataset):

    def __init__(self, root_path='../dataset', mode='train',image_size=224, is_train=True, remove_duplicate=True,
                 is_sample_positive=0.5):
        super(Fakeddit_dataset, self).__init__()
        self.remove_duplicate = remove_duplicate
        self.is_train = is_train
        self.root_path = root_path
        self.is_sample_positive = is_sample_positive
        self.index = 0
        self.label_dict = []
        self.image_size = image_size
        self.image_path = self.root_path+'/Fakeddit/images/images'
        print(self.image_path)
        list_dirs = os.listdir(_fnd_read_path(self.image_path))
        self.image_listdir = {}
        for list_dir in list_dirs:
            idx = list_dir.rfind('.')
            self.image_listdir[list_dir[:idx]] = list_dir

        train_path = root_path + '/Fakeddit/multimodal_' + mode +"_cleaned"+'.tsv'
        test_path = root_path+"/Fakeddit/multimodal_validate_cleaned.tsv"
        df = pd.read_csv(_fnd_read_path(train_path), sep="\t")
        df = df.replace(np.nan, '', regex=True)
        df.fillna('', inplace=True)
        
        self.IMAGE_DICT, self.IMAGE_DICT_KEYS = {}, []
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
            record['content'] = []
            record['content'].append(content)
            # Preserve each record when duplicate handling is disabled.
            if not self.remove_duplicate:
                if label == 0:
                    negative_records.append(record)
                else:
                    positive_records.append(record)
                self.label_dict.append(record)
            else:
                key = images_name+"_"+str(label)
                if key not in self.IMAGE_DICT:
                    if label == 0:
                        negative_records.append(record)
                    else:
                        positive_records.append(record)
                    
                else:
                    if self.is_sample_positive < 1 and label == 1 and random.random() > self.is_sample_positive:
                        continue
                    record_previous = self.IMAGE_DICT[key]
                    record_previous['content'].append(content)
        min_len = min(len(positive_records), len(negative_records))
        if mode == "train":
            self.label_dict = positive_records + negative_records
        else:
            self.label_dict = positive_records + negative_records

        assert len(self.label_dict)!=0, 'Error: GT path is empty.'

    def __getitem__(self, index):

        GT_size = self.image_size

        # Select an image for this record.
        record = self.label_dict[index]
        images, label, content_list = record['images'], record['label'], record['content']

        idx_rand = random.randint(0, len(content_list) - 1)
        content = content_list[idx_rand]

        imgs = images.split(',')
        GT_path = imgs[np.random.randint(0, len(imgs))]

        try:
            is_found_image = False
            if GT_path in self.image_listdir:
                GT_path = self.image_listdir[GT_path]
                is_found_image = True

            GT_path = "{}/{}".format(self.image_path,GT_path)
            img_GT = read_img(GT_path)
        except Exception:
            raise IOError("Load {} Error {}".format(GT_path, record['images']))

        img_GT = channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]

        # Resize without cropping.
        img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                            interpolation=cv2.INTER_LINEAR)

        orig_height, orig_width, _ = img_GT.shape
        H, W, _ = img_GT.shape

        # Reverse the channel order and convert HWC to a CHW tensor.
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        return (content, _, _), (img_GT, _, label, _, _),_

    def __len__(self):
        return len(self.label_dict)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

if __name__ == "main":
    train_dataset = Fakeddit_dataset(root_path="../dataset")
    print(train_dataset)

