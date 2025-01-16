# datasets.py
import glob, os, sys, pickle
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

import config

class XRaysTrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load full dataframe
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.make_pkl_dir(config.pkl_dir_path)

        # Load or create train_val_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path)):
            self.train_val_df = self.get_train_val_df()
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('{}: dumped'.format(config.train_val_df_pkl_path))
        else:
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
            print('{}: loaded'.format(config.train_val_df_pkl_path))

        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()

        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path)):
            with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.new_df = self.train_val_df.iloc[self.the_chosen, :]

    def get_all_classes(self):
        """ Return all_classes even if dataset is a subset """
        return self.all_classes

    def resample(self):
        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
        self.new_df = self.train_val_df.iloc[self.the_chosen, :]

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_train_val_df(self):
        train_val_list = self.get_train_val_list()
        train_val_df = pd.DataFrame()
        for i in tqdm(range(self.df.shape[0])):
            filename = os.path.basename(self.df.iloc[i, 0])
            if filename in train_val_list:
                train_val_df = pd.concat([train_val_df, self.df.iloc[i:i+1, :]], ignore_index=True)
        return train_val_df

    def __getitem__(self, index):
        row = self.new_df.iloc[index, :]
        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')

        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def choose_the_indices(self):
        max_examples_per_class = 5000
        the_chosen = []
        all_classes = {}

        for i in tqdm(list(np.random.choice(range(len(self.train_val_df)), len(self.train_val_df), replace=False))):
            temp = str.split(self.train_val_df.iloc[i, :]['Finding Labels'], '|')

            if 'Hernia' in temp:
                the_chosen.append(i)
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        all_classes[t] += 1
                continue

            if len(temp) > 1:
                bool_lis = [False] * len(temp)
                for idx, t in enumerate(temp):
                    if t in all_classes:
                        if all_classes[t] < max_examples_per_class:
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                if sum(bool_lis) == len(temp):
                    the_chosen.append(i)
                    for t in temp:
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class:
                            all_classes[t] += 1
                            the_chosen.append(i)

        # Assign the all_classes as an instance attribute for easy access
        self.all_classes = sorted(list(all_classes))
        self.all_classes_dict = all_classes  # Optional: keeping the dict as well if needed

        return the_chosen, self.all_classes, self.all_classes_dict

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]
        df['Image Index'] = df['image_links'].apply(lambda x: x[len(x) - 16:len(x)])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Finding Labels']]
        return merged_df

    def get_train_val_list(self):
        with open(os.path.join('data', 'NIH Chest X-rays', 'train_val_list.txt'), 'r') as f:
            train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.new_df)


# Test Dataset Class
class XRaysTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)

        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle)

        if not os.path.exists(os.path.join(config.pkl_dir_path, config.test_df_pkl_path)):
            self.test_df = self.get_test_df()
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)

    def __getitem__(self, index):
        row = self.test_df.iloc[index, :]
        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')

        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]
        df['Image Index'] = df['image_links'].apply(lambda x: x[len(x) - 16:len(x)])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Finding Labels']]
        return merged_df

    def get_test_df(self):
        test_list = self.get_test_list()
        test_df = pd.DataFrame()
        for i in tqdm(range(self.df.shape[0])):
            filename = os.path.basename(self.df.iloc[i, 0])
            if filename in test_list:
                test_df = pd.concat([test_df, self.df.iloc[i:i+1, :]], ignore_index=True)
        return test_df

    def get_test_list(self):
        with open(os.path.join('data', 'NIH Chest X-rays', 'test_list.txt'), 'r') as f:
            test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)
