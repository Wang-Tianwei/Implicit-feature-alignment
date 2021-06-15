# coding:utf-8
import torch
import torch.optim as optim
import os
from IAM_data_loader import *
from torchvision import transforms
from ExCTC import *

global_cfgs = {
    'state': 'Test',
    'epoch': 40,
    'show_interval': 50,
    'test_interval': 500,
}
data_args = {
    'AUG'       : True,
    'Scale'     : False,
    'Train_Batch': 24, 
    'HEIGHT'    : 80,
    'WIDTH'     : 1200,
    'FINAL_LEN' : 150,
    'DICT'      : '!#"\'&)(+*-,/.1032547698;:?ACBEDGFIHKJMLONQPSRUTWVYXZacbedgfihkjmlonqpsrutwvyxz|',
}
dataset_cfgs = {    
    'dataset_train': IAMSynthesisDataset,
    'dataset_train_args': {
        'ImgList': 'data/train_list.txt',
        'LineFolder': 'path to lines folder',
        'WordFolder': 'path to words folder',
        'LabelFile': 'data/labels.txt',
        'WordLabel': 'data/words.txt',
        'length': 32 * 2000,
    },
    'dataloader_train': {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0,
        'collate_fn': PadCollate(),
    },

    'dataset_test': IAMDataset_Fullpage,
    'dataset_test_args': {
        'ImgList': 'demo_data/demo_img_list.txt',
        'ImgFolder': 'demo_data/pargs/',
        'LabelFile': 'data/labels.txt',
    },
    'dataloader_test': {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0,
    },

    'dataset_val': IAMDataset,
    'dataset_val_args': {
        'ImgList': 'demo_data/demo_img_list.txt',
        'ImgFolder': 'demo_data/lines/',
        'LabelFile': 'data/labels.txt',
    },
    'dataloader_val': {
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 0,
        'collate_fn': PadCollate(),
    },
}

net_cfgs = {
    'model': ExCTC,
    'model_args': {
        'nClass': len(data_args['DICT'])+1,
    },

    'init_state_dict': 'models/IAM_ExCTC.pth',
}

optimizer_cfgs = {
    # optim for FE
    'optimizer_0': optim.Adam,
    'optimizer_0_args':{
        'lr': 1e-3,
    },

    'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_0_scheduler_args': {
        'milestones': [15, 20],
        'gamma': 0.1,
    },
}

saving_cfgs = {
    'saving_iter_interval': 1000,
    'saving_epoch_interval': 3,

    'saving_path': 'models/ExCTC/exp1_',
}

def mkdir(path_):
    paths = path_.split('/')
    command_str = 'mkdir '
    for i in range(0, len(paths) - 1):
        command_str = command_str + paths[i] + '/'
    command_str = command_str[0:-1]
    os.system(command_str)

def showcfgs(s):
    for key in s.keys():
        print(key , s[key])
    print('')

mkdir(saving_cfgs['saving_path'])