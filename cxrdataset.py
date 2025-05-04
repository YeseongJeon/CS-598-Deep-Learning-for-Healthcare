#!/usr/bin/env python3
import os

import numpy
import torch
import pandas
import sklearn.model_selection
from torchvision import transforms
from PIL import Image, ImageFile
import random


ImageFile.LOAD_TRUNCATED_IMAGES = True

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def _get_patient_id(path):
    return path.split('/')[2]


def _nih_get_patient_id(path):

    image_path = path.split('/')[4]
    patient_id = image_path.split('_')[0]

    return patient_id


def _mimic_get_patient_id(path):

    patient_id = path.split('/')[0].replace("p", "")

    return patient_id


def _get_unique_patient_ids(dataframe, dataset):
    ids = list(dataframe.index)
    if dataset == "NIH":
        ids = [_nih_get_patient_id(i) for i in ids]
    elif dataset == "MIMIC":

        ids = [_mimic_get_patient_id(i) for i in ids]
    else:
        ids = [_get_patient_id(i) for i in ids]

    ids = list(set(ids))
    ids.sort()

    return ids


def grouped_split(dataframe, random_state=None, test_size=0.05, dataset="NIH"):
    '''
    Split a dataframe such that patients are disjoint in the resulting folds.
    The dataframe must have an index that contains strings that may be processed
    by _get_patient_id to return the unique patient identifiers.
    '''

    print(dataframe)

    groups = _get_unique_patient_ids(dataframe, dataset)

    print(f"Number of samples: {len(groups)}")

    traingroups, testgroups = sklearn.model_selection.train_test_split(
        groups,
        random_state=random_state,
        test_size=test_size)
    traingroups = set(traingroups)
    testgroups = set(testgroups)

    trainidx = []
    testidx = []

    for idx, row in dataframe.iterrows():
        if dataset == "NIH":
            patient_id = _nih_get_patient_id(idx)
        elif dataset == "MIMIC":
            patient_id = _mimic_get_patient_id(idx)
        else:
            patient_id = _get_patient_id(idx)
        # print(patient_id)
        if patient_id in traingroups:
            trainidx.append(idx)
        elif patient_id in testgroups:
            testidx.append(idx)
    num_trainids = len(trainidx)
    num_testids = len(testidx)

    traindf = dataframe.loc[dataframe.index.isin(trainidx), :]
    testdf = dataframe.loc[dataframe.index.isin(testidx), :]
    return traindf, testdf


class CXRDataset(torch.utils.data.Dataset):
    '''
    Base class for chest radiograph datasets.
    '''
    # define torchvision transforms as class attribute
    _transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    _transforms['test'] = _transforms['val']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if (self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return (image, label, self.df.index[idx], ['None'])

    def get_all_labels(self):
        '''
        Return a numpy array of shape (n_samples, n_dimensions) that includes 
        the ground-truth labels for all samples.
        '''
        ndim = len(self.labels)
        nsamples = len(self)
        output = numpy.zeros((nsamples, ndim))
        for isample in range(len(self)):
            output[isample] = self[isample][1]
        return output


class CheXpertDataset(CXRDataset):
    """
    Dataset to load the CheXpert X-Ray images dataset from the original paper. 
    We didn't use this since we didn't utilize the CheXpert dataset.
    """

    def __init__(
            self,
            fold,
            include_lateral=False,
            random_state=30493):
        '''
        Create a dataset of the CheXPert images for use in a PyTorch model.

        Args:
            fold (str): The shard of the CheXPert data that the dataset should
                contain. One of either 'train', 'val', or 'test'. The 'test'
                fold corresponds to the images specified in 'valid.csv' in the 
                CheXPert data, while the the 'train' and 'val' folds
                correspond to disjoint subsets of the patients in the 
                'train.csv' provided with the CheXpert data.
            random_state (int): An integer used to see generation of the 
                train/val split from the patients specified in the 'train.csv'
                file provided with the CheXpert dataset. Used to ensure 
                reproducability across runs.
            include_lateral (bool): If True, include the lateral radiograph
                views in the dataset. If False, include only frontal views.
        '''

        self.transform = self._transforms[fold]
        self.path_to_images = "../data/CheXpert/"
        self.fold = fold

        # Load files containing labels, and perform train/valid split if necessary
        if fold == 'train' or fold == 'val':
            trainvalpath = os.path.join(
                self.path_to_images,
                'CheXpert-v1.0-small/train.csv')
            self.df = pandas.read_csv(trainvalpath)
            self.df.set_index("Path", inplace=True)

            if not include_lateral:
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            train, val = grouped_split(
                self.df,
                random_state=random_state,
                test_size=0.05)
            if fold == 'train':
                self.df = train
            else:
                self.df = val
        elif fold == 'test':
            testpath = os.path.join(
                self.path_to_images,
                'CheXpert-v1.0-small/valid.csv')
            self.df = pandas.read_csv(testpath)
            self.df.set_index("Path", inplace=True)
            if not include_lateral:
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        else:
            raise ValueError("Invalid fold: {:s}".format(str(fold)))

        self.labels = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if (self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        appa = self.df['AP/PA'][idx] == 'AP'

        return (image, label, self.df.index[idx], appa)

    def get_all_view_labels(self):
        '''
        Return a numpy array of shape (n_samples, n_dimensions) that includes 
        the ground-truth labels for all samples.
        '''
        ndim = 1
        nsamples = len(self)
        output = numpy.zeros((nsamples, 1))
        for isample in range(len(self)):
            output[isample] = self.df['AP/PA'][isample] == 'AP'
        return output


class NIHDataset(CXRDataset):
    """
    Dataset to load the NIH X-Ray images dataset. 
    We followed the logic from the CheXpert Dataset implementation from the original paper and configued it for our NIH Dataset.
    """

    def __init__(
            self,
            fold,
            include_lateral=False,
            random_state=31242):
        '''
        
        Args:
            fold (str): The shard of the CheXPert data that the dataset should
                contain. One of either 'train', 'val', or 'test'. The 'test'
                fold corresponds to the images specified in 'valid.csv' in the 
                CheXPert data, while the the 'train' and 'val' folds
                correspond to disjoint subsets of the patients in the 
                'train.csv' provided with the CheXpert data.
            random_state (int): An integer used to see generation of the 
                train/val split from the patients specified in the 'train.csv'
                file provided with the CheXpert dataset. Used to ensure 
                reproducability across runs.
            include_lateral (bool): If True, include the lateral radiograph
                views in the dataset. If False, include only frontal views.
        '''

        self.transform = self._transforms[fold]
        self.path_to_images = "data/NIH/"
        self.fold = fold

        # Load files containing labels, and perform train/valid split if necessary
        if fold == 'train' or fold == 'val':
            trainvalpath = os.path.join(
                self.path_to_images,
                'nih_train.csv')
            self.df = pandas.read_csv(trainvalpath)
            self.df.set_index("Path", inplace=True)

            if not include_lateral:
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            train, val = grouped_split(
                self.df,
                random_state=random_state,
                test_size=0.05)
            if fold == 'train':
                self.df = train
            else:
                self.df = val
        elif fold == 'test':
            testpath = os.path.join(
                self.path_to_images,
                'nih_valid.csv')
            self.df = pandas.read_csv(testpath)
            self.df.set_index("Path", inplace=True)
            if not include_lateral:
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        else:
            raise ValueError("Invalid fold: {:s}".format(str(fold)))

        self.labels = ['Edema',
                       'Fibrosis',
                       'Nodule',
                       'Infiltration',
                       'Pneumothorax',
                       'Pleural_Thickening',
                       'Pneumonia',
                       'Atelectasis',
                       'Mass',
                       'Cardiomegaly',
                       'Emphysema',
                       'Effusion',
                       'Hernia',
                       'Consolidation']

    def __getitem__(self, idx):

        # image = Image.open(
        #     os.path.join(
        #         self.path_to_images,
        #         self.df.index[idx]))
        image = Image.open(self.df.index[idx])

        image = image.convert('RGB')

        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if (self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        appa = self.df['AP/PA'][idx] == 'AP'

        return (image, label, self.df.index[idx], appa)

    def get_all_view_labels(self):
        '''
        Return a numpy array of shape (n_samples, n_dimensions) that includes 
        the ground-truth labels for all samples.
        '''
        ndim = 1
        nsamples = len(self)
        output = numpy.zeros((nsamples, 1))
        for isample in range(len(self)):
            output[isample] = self.df['AP/PA'][isample] == 'AP'
        return output


class MIMICDataset(CXRDataset):
    """
    Dataset to load the MIMIC X-Ray images dataset.
    We modified the original implementation to fit our data organization and also added the AP/PA labels.
    """
    def __init__(
            self,
            fold,
            include_lateral=False,
            random_state=30493):
        '''
        Create a dataset of the MIMIC-CXR images for use in a PyTorch model.

        Args:
            fold (str): The shard of the MIMIC-CXR data that the dataset should
                contain. One of either 'train', 'val', or 'test'.
            random_state (int): An integer used to see generation of the 
                train/val split
            include_lateral (bool): If True, include the lateral radiograph
                views in the dataset. If False, include only frontal views.
        '''
        print("Running MIMIC DATASET")
        self.transform = self._transforms[fold]
        self.path_to_images = "./data/MIMIC/"
        self.fold = fold

        # Load files containing labels, and perform train/valid split if necessary
        if fold == 'train' or fold == 'val':
            trainvalpath = os.path.join(self.path_to_images, 'mimic_train.csv')
            self.df = pandas.read_csv(trainvalpath)
            self.df.set_index("path", inplace=True)
            if not include_lateral:
                self.df = self.df[self.df['ViewPosition'].isin(['AP', 'PA'])]
            train, val = grouped_split(
                self.df,
                random_state=random_state,
                test_size=0.05,
                dataset="MIMIC")
            if fold == 'train':
                self.df = train
            else:
                self.df = val
        elif fold == 'test':
            testpath = os.path.join(self.path_to_images, 'mimic_valid.csv')
            self.df = pandas.read_csv(testpath)
            self.df.set_index("path", inplace=True)
            if not include_lateral:
                self.df = self.df[self.df['ViewPosition'].isin(['AP', 'PA'])]
        else:
            raise ValueError("Invalid fold: {:s}".format(str(fold)))

        self.labels = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if (self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)
        appa = self.df['ViewPosition'].iloc[idx] == 'AP'
        return (image, label, self.df.index[idx], appa)

    def get_all_view_labels(self):
        '''
        Return a numpy array of shape (n_samples, 1) for AP/PA labels.
        '''
        ndim = 1
        nsamples = len(self)
        output = numpy.zeros((nsamples, 1))
        for isample in range(len(self)):
            view_val = self.df['ViewPosition'].iloc[isample]
            output[isample] = view_val == 'AP'
        return output
