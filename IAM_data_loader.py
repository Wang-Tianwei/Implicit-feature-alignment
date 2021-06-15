# CursorOCRDataset
# CursorTestOCRDataset
# KeyOCRDataset
# ImageDataset
# BatchOCRDataset


from torch.utils.data import Dataset, DataLoader
import time
import math
import numpy as np
import torchvision.transforms as transforms
import six
from PIL import Image
import os
import torch
import lmdb
import cv2
import sys
import cfgs
from augment import distort, stretch, perspective
import random

def pad_tensor(vec, pad, dim, istrain):
    vec = torch.tensor(vec)
    pad_size_front = list(vec.shape)
    if istrain:
        pad_size_front[dim] = int((pad - vec.size(dim)) * random.random())
    else:
        pad_size_front[dim] = int((pad - vec.size(dim)) * 0.5)
    pad_size_back = list(vec.shape)
    pad_size_back[dim] = pad - vec.size(dim) - pad_size_front[dim]
    return torch.cat([torch.ones(*pad_size_front).type_as(vec.data), \
                      vec, torch.ones(*pad_size_back).type_as(vec.data)], dim=dim)

class PadCollate:
    def __init__(self, dim=0, istrain=True):
        self.dim = dim
        self.istrain = istrain

    def pad_collate(self, batch):
        # find longest sequence
        max_W = max(map(lambda x: x['image'].shape[2], batch))
        max_W = max_W + 8 - max_W % 8
        for i in range(0, len(batch)):
            batch[i]['image'] = pad_tensor(batch[i]['image'], pad=max_W, dim=2, istrain=self.istrain)
        # stack all
        output = {}
        for k in batch[0].keys():
            if type(batch[0][k]) == torch.Tensor:
                output[k] = torch.stack([batch[i][k] for i in range(0, len(batch))], dim=0)
            elif type(batch[0][k]) == str:
                output[k] = [batch[i][k] for i in range(0, len(batch))]
        return output

    def __call__(self, batch):
        return self.pad_collate(batch)

class LineGenerate():

    def __init__(self, ImgList, ImgFolder, LabelFile, conH, conW, training=False):
        self.training = training
        self.aug = cfgs.data_args['AUG']

        self.conH = conH
        self.conW = conW
        standard = []
        with open(ImgList) as f:
            for line in f.readlines():
                standard.append(line.strip('\n'))
        self.image = []
        self.label = []

        count = 0
        with open(LabelFile) as f:
            for line in f.readlines():
                elements = line.split()
                pth_ele = elements[0].split('-')
                line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
                if line_tag in standard:
                    pth = ImgFolder + '%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])
                    img= cv2.imread(pth, 0) #see channel and type
                    if img is not None:
                        self.image.append(img)
                        self.label.append(elements[-1])
                        count += 1
                    else:
                        print(pth)
                        print('line error')
        self.len = count
        self.idx = 0

    def get_len(self):
        return self.len

    def generate_line(self):
        if self.training:
            idx = np.random.randint(self.len)
            image = self.image[idx]
            label = self.label[idx]
        else:
            idx = self.idx
            image = self.image[idx]
            label = self.label[idx]
            self.idx += 1

        if self.idx == self.len:
            self.idx -= self.len

        h,w = image.shape

        if self.training and cfgs.data_args['Scale']:
            dsth = np.random.randint(self.conH//3, self.conH+1)
        else:
            dsth = self.conH

        if float(h) / w > float(dsth) / self.conW:
            newW = int(w * dsth / float(h))
            imageN = np.ones((self.conH, newW))*255
            if self.training:
                beginH = np.random.randint(0, int(abs(self.conH-dsth))+1)
            else:
                beginH = int(abs(self.conH-dsth)/2)
            image = cv2.resize(image, (newW, dsth))
            image = image.astype('uint8')
            imageN[beginH:beginH+dsth] = image
        else:
            newH = int(h * self.conW / float(w))
            imageN = np.ones((self.conH, self.conW))*255
            if self.training:
                beginH = np.random.randint(0, int(abs(self.conH-newH))+1)
            else:
                beginH = int(abs(self.conH-newH)/2)
            image = cv2.resize(image, (self.conW, newH))
            image = image.astype('uint8')
            imageN[beginH:beginH+newH] = image

        label = self.label[idx]

        if self.training and cfgs.data_args['AUG'] and imageN.shape[1] > 50:
            if np.random.rand() < 0.5:
                imageN = distort(imageN, max(2, imageN.shape[1]//imageN.shape[0]))
            if np.random.rand() < 0.5:
                imageN = stretch(imageN, max(2, imageN.shape[1]//imageN.shape[0]))      
            if np.random.rand() < 0.5:
                imageN = perspective(imageN)
            if np.random.rand() < 0.01:
                cv2.imwrite('augsample/%04d.jpg' % np.random.randint(10000), imageN)

        if torch.rand(1) < 0.01:
            cv2.imwrite("line_generate/%s.jpg" % label, imageN)

        imageN = imageN.astype('float32')
        imageN = (imageN-127.5)/127.5
        return imageN, label

import json


class IAMDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ImgList, ImgFolder, LabelFile, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.training = False     

        self.conW = cfgs.data_args['WIDTH']
        self.conH = cfgs.data_args['HEIGHT']
        self.LG = LineGenerate(ImgList, ImgFolder, LabelFile, self.conH, self.conW, self.training)

        self.dict = {}
        self.alphabet = cfgs.data_args['DICT']
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

    def __len__(self):
        return self.LG.get_len()


    def __getitem__(self, idx):
        
        imageN, label = self.LG.generate_line()

        try:
            text = [self.dict[char] for char in label]
        except:
            pdb.set_trace()

        label = np.zeros((1, cfgs.data_args['FINAL_LEN']))-1
        label[0,:len(text)]=text
        label = label.astype('int')

        try:
            imageN = imageN.reshape(1, imageN.shape[0], imageN.shape[1])
            sample = {'image': torch.from_numpy(imageN), 'label': torch.from_numpy(label)}
        except:
            pdb.set_trace()

        return sample  

import scipy.signal
def resize_ratio(img, target_h = 96.):
    img = 255 - img
    h, w = img.shape

    spec = img.sum(1)
    max_value = spec.max()

    peaks, value = scipy.signal.find_peaks(spec, prominence=10000)
    peaks_w = scipy.signal.peak_widths(spec, peaks, rel_height=0.98)
    
    mean_h = np.array(peaks_w[0]).mean()
    aspect_ratio = target_h / mean_h
    
    return min(0.65, aspect_ratio)

class IAMDataset_Fullpage(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ImgList, ImgFolder, LabelFile):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.training = False  

        self.conW = cfgs.data_args['WIDTH']
        self.conH = cfgs.data_args['HEIGHT']

        self.image = []
        self.label = []

        IAMLine = open(LabelFile).readlines()
        IAMLine_dict = {}
        for l in IAMLine:
            lsplit = l.split(' ')
            filename = '-'.join(lsplit[0].split('-')[:-1])
            if filename not in IAMLine_dict:
                IAMLine_dict[filename] = lsplit[-1].strip('\n')
            else:
                IAMLine_dict[filename] += '|'
                IAMLine_dict[filename] += lsplit[-1].strip('\n')

        for file in open(ImgList).readlines():
            file = file.strip('\n')
            imagepath = ImgFolder + file + '.png'
            self.image.append(cv2.imread(imagepath, 0))
            self.label.append(IAMLine_dict[file])

        self.dict = {}
        self.alphabet = cfgs.data_args['DICT']
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

        self.len = len(self.image)
        self.idx = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.training:
            idx = np.random.randint(self.len)
            image = self.image[idx]
            label = self.label[idx]
        else:
            idx = self.idx
            image = self.image[idx]
            label = self.label[idx]
            self.idx += 1
            
        label = label.replace(' ', '|')

        if self.idx == self.len:
            self.idx -= self.len 

        try:
            text = [self.dict[char] for char in label]
        except:
            pdb.set_trace()

        label = np.zeros(len(text))-1
        label[:len(text)] = text
        label = label.astype('int')

        image = np.array(image)
        image = image.astype('float32')

        # aspect_ratio = resize_ratio(image)
        aspect_ratio = 0.54
        # print(aspect_ratio)
        imageN = (image - 127.5) / 127.5
        wnew = int(imageN.shape[1] * aspect_ratio)
        hnew = int(imageN.shape[0] * aspect_ratio)
        wnew = wnew + 8 - wnew % 8
        hnew = hnew + 8 - hnew % 8
        imageN = cv2.resize(imageN, (wnew, hnew))

        conH, conW = imageN.shape
        imageN = imageN.reshape(1, conH, conW)

        sample = {'image': torch.from_numpy(imageN), 'label': torch.from_numpy(label)}
        return sample    
        

class WordGenerate():
    def __init__(self, ImgList, WordFolder, WordLabel, conH, conW):

        print('load word')
        self.conH = conH
        self.conW = conW
        standard = []
        with open(ImgList) as f:
            for line in f.readlines():
                standard.append(line.strip('\n'))
        self.image = []
        self.label = []

        self.eng_chars = 'ACBEDGFIHKJMLONQPSRUTWVYXZacbedgfihkjmlonqpsrutwvyxz'
        # IAMWord = '/media/dlvc/StoringDisk1/kpi/0001_code/0003_exctciam/0000_data/words.txt'
        count = 0
        with open(WordLabel) as f:
            for line in f.readlines():
                elements = line.split()
                pth_ele = elements[0].split('-')
                line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
                if line_tag in standard:
                    pth = WordFolder + '%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])
                    img= cv2.imread(pth, 0) #see channel and type
                    if img is not None and elements[1] == 'ok':
                        labeltmp = ''.join(elements[8:])
                        used = True
                        for c in labeltmp:
                            if c not in self.eng_chars:
                                used = False
                                break
                        if used:
                            self.image.append(img)
                            self.label.append(labeltmp)
                            count += 1
                    else:
                        # print(pth)
                        # print('word error')
                        continue;
        print('total word: %d' % count)
        self.len = count

    def get_len(self):
        return self.len

    def word_generate(self):
        ## one word
        label = ''
        iseng_chars = True
        if cfgs.data_args['Scale']:
            scale_ratio = np.random.rand() * 2 + 1
        else:
            scale_ratio = 1

        if np.random.rand() < 0.3:
            idx = np.random.randint(self.len)
            image = self.image[idx]
            # iseng_chars = self.iseng_chars[idx]
            image = cv2.resize(image, (int(image.shape[1]//scale_ratio), int(image.shape[0]//scale_ratio)))
            h, w = image.shape
            dsth = self.conH
            if h >= dsth:
                newW = int(w * dsth / float(h))
                imageN = np.ones((self.conH, newW))*255
                beginH = np.random.randint(0, int(abs(self.conH-dsth))+1)
                image = cv2.resize(image, (newW, dsth))
                if cfgs.data_args['AUG'] and image.shape[1] > 50:
                    if np.random.rand() < 0.5:
                        image = distort(image, max(2, image.shape[1]//image.shape[0]))
                    if np.random.rand() < 0.5:
                        image = stretch(image, max(2, image.shape[1]//image.shape[0]))      
                    if np.random.rand() < 0.5:
                        image = perspective(image)
                image = image.astype('uint8')
                imageN[beginH:beginH+dsth] = image
            else:
                imageN = np.ones((self.conH, w))*255
                if iseng_chars:
                    beginH = np.random.randint(self.conH-h)
                else:
                    beginH = (self.conH-h)//2
                if cfgs.data_args['AUG'] and image.shape[1] > 50:
                    if np.random.rand() < 0.5:
                        image = distort(image, max(2, image.shape[1]//image.shape[0]))
                    if np.random.rand() < 0.5:
                        image = stretch(image, max(2, image.shape[1]//image.shape[0]))      
                    if np.random.rand() < 0.5:
                        image = perspective(image)
                imageN[beginH:beginH+h] = image
            label = self.label[idx]
        ## one line

        else:
            endW = 0
            dstw = np.random.randint(self.conW//2, self.conW)
            imageN = np.ones((self.conH, dstw))*255
            while True:
                idx = np.random.randint(self.len)
                image = self.image[idx]
                # iseng_chars = self.iseng_chars[idx]
                image = cv2.resize(image, (int(image.shape[1]//scale_ratio), int(image.shape[0]//scale_ratio)))
                h, w = image.shape
                if h >= self.conH:
                    newW = int(w * self.conH / float(h))
                    if endW + newW > dstw:
                        break;
                    image = cv2.resize(image, (newW, self.conH))
                    if cfgs.data_args['AUG'] and image.shape[1] > 50:
                        if np.random.rand() < 0.5:
                            image = distort(image, max(2, image.shape[1]//image.shape[0]))
                        if np.random.rand() < 0.5:
                            image = stretch(image, max(2, image.shape[1]//image.shape[0]))      
                        if np.random.rand() < 0.5:
                            image = perspective(image)
                    imageN[:,endW:endW+newW] = image
                    endW += np.random.randint(25)+25+newW
                else:
                    if endW + w > dstw:
                        break;
                    if iseng_chars:
                        beginH = np.random.randint(self.conH-h)
                    else:
                        beginH = (self.conH-h)//2
                    if cfgs.data_args['AUG'] and image.shape[1] > 50:
                        if np.random.rand() < 0.5:
                            image = distort(image, max(2, image.shape[1]//image.shape[0]))
                        if np.random.rand() < 0.5:
                            image = stretch(image, max(2, image.shape[1]//image.shape[0]))      
                        if np.random.rand() < 0.5:
                            image = perspective(image)
                    imageN[beginH:beginH+h, endW:endW+w] = image
                    endW += np.random.randint(25)+25+w
                if label == '':
                    label = self.label[idx]
                else:
                    label = label + '|' + self.label[idx]

        label = label

        if torch.rand(1) < 0.01:
            cv2.imwrite("word_generate/%s.jpg" % label, imageN)

        imageN = imageN.astype('float32')
        imageN = (imageN-127.5)/127.5
        return imageN, label        

class IAMSynthesisDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ImgList, LineFolder, WordFolder, LabelFile, WordLabel, length, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.conW = cfgs.data_args['WIDTH']
        self.conH = cfgs.data_args['HEIGHT']

        # self.SynLG = SynLineGenerate(self.conH, self.conW)
        self.WG = WordGenerate(ImgList, WordFolder, WordLabel, self.conH, self.conW)
        self.LG = LineGenerate(ImgList, LineFolder, LabelFile, self.conH, self.conW, True)

        self.dict = {}
        self.alphabet = cfgs.data_args['DICT']
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

        self.length = length

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        sd = np.random.rand()
        if sd < 0.5:
            imageN, label = self.WG.word_generate()        
        # elif sd < 0.6:
        #     imageN, label = self.SynLG.generate_line()        
        else:
            imageN, label = self.LG.generate_line()

        try:
            text = [self.dict[char] for char in label]
        except:
            pdb.set_trace()

        label = np.zeros((1,cfgs.data_args['FINAL_LEN']))-1
        label[0,:len(text)]=text
        label = label.astype('int')
        
        try:
            imageN = imageN.reshape(1, imageN.shape[0], imageN.shape[1])
            sample = {'image': torch.from_numpy(imageN), 'label': torch.from_numpy(label)}
        except:
            pdb.set_trace()

        return sample   
