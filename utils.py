import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import json
import edit_distance as ed
import pdb
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont
import sys
import math
import time


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
    
def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))

def cal_distance(label_list, pre_list):
    y = ed.SequenceMatcher(a = label_list, b = pre_list)
    yy = y.get_opcodes()
    insert = 0
    delete = 0
    replace = 0
    for item in yy:
        if item[0] == 'insert':
            insert += item[-1]-item[-2]
        if item[0] == 'delete':
            delete += item[2]-item[1]
        if item[0] == 'replace':
            replace += item[-1]-item[-2]  
    distance = insert+delete+replace     
    return distance, (delete, replace, insert)  

class Loss_counter():
    def __init__(self):
        self.total_iters = 0.
        self.loss_sum = 0
    
    def add_iter(self, loss):
        self.total_iters += 1
        self.loss_sum += float(loss)

    def clear(self):
        self.total_iters = 0
        self.loss_sum = 0
    
    def get_loss(self):
        loss = self.loss_sum / self.total_iters if self.total_iters > 0 else 0
        self.total_iters = 0
        self.loss_sum = 0
        return loss

def cal_distance_wer(label_list, pre_list, alphabet):
    label_chn = ''.join([alphabet[cha-1] for cha in label_list])
    pred_chn = ''.join([alphabet[cha-1] for cha in pre_list])    
    label_chn_word = []
    strtmp = ''
    for s in label_chn:
        if s == '|':
            label_chn_word.append(strtmp)
            strtmp = ''
        else:
            strtmp += s

    pred_chn_word = []
    strtmp = ''
    for s in pred_chn:
        if s == '|':
            pred_chn_word.append(strtmp)
            strtmp = ''
        else:
            strtmp += s    

    wordlen = len(label_chn_word)

    y = ed.SequenceMatcher(a = label_chn_word, b = pred_chn_word)
    yy = y.get_opcodes()
    insert = 0
    delete = 0
    replace = 0
    for item in yy:
        if item[0] == 'insert':
            insert += item[-1]-item[-2]
        if item[0] == 'delete':
            delete += item[2]-item[1]
        if item[0] == 'replace':
            replace += item[-1]-item[-2]  
    distance = insert+delete+replace    
    return distance, wordlen

class CTC_AR_counter():
    def __init__(self):
        self.delete_error = 0.
        self.replace_error = 0.
        self.insert_error = 0.
        self.character_total = 0
    def clear(self):
        self.delete_error = 0.
        self.replace_error = 0.
        self.insert_error = 0.
        self.character_total = 0
    def add_iter(self, output, labels):
        B = output.size(1)
        for i in range(0, B):
            raw_pred = output[:, i, :].topk(1)[1].squeeze().tolist()
            pred = [raw_pred[j] for j in range(1, len(raw_pred)) if raw_pred[j] != raw_pred[j-1]]
            pred = [_-1 for _ in pred if _ > 0]
            label_i = labels[i].tolist()
            label_i = [_ for _ in label_i if _ > 0]
            distance, (delete, replace, insert) = cal_distance(pred, label_i)
            self.character_total += len(label_i)
            self.delete_error += delete
            self.insert_error += insert
            self.replace_error += replace
    def show(self, iter_id=0, clear=False, start_time=None):
        CR = 1 - (self.delete_error+self.replace_error) / self.character_total
        AR = 1 - (self.delete_error+self.replace_error+self.insert_error) / self.character_total
        if start_time:
            print('Test : %10s iter_id: %6d, CR: %4f  AR: %4f' % 
                (timeSince(start_time), iter_id, CR, AR))
        else:
            print('iter_id: %6d, CR: %4f  AR: %4f' % 
                (iter_id, CR, AR))
        self.clear()
            # pdb.set_trace()

class Sequence(nn.Module):
    def __init__(self, alphabet):
        super(Sequence, self).__init__()
        self.count_n = 0
        self.showing = False
        self.softmax = None
        self.label = None
        self.image = None
        
        self.delete_error = 0.
        self.replace_error = 0.
        self.insert_error = 0.
        self.word_error = 0.
        self.pred_character_total = 0
        self.character_total = 0
        self.word_total = 0
        self.images = 0
        self.alphabet = alphabet
        
    def clear(self):
        self.delete_error = 0.
        self.replace_error = 0.
        self.insert_error = 0.
        self.word_error = 0.
        self.pred_character_total = 0
        self.character_total = 0
        self.word_total = 0
        self.images = 0

    def show(self, iter_id=0, clear=False, start_time=None):
        CR = 1 - (self.delete_error+self.replace_error) / self.character_total
        AR = 1 - (self.delete_error+self.replace_error+self.insert_error) / self.character_total
        WER = self.word_error/self.word_total
        if start_time:
            print('Test : %10s iter_id: %6d, CER: %4f  WER: %4f \n %10s delete: %4d, replace: %4d, insert: %4d, gt_len: %4d, pred_len: %4d, total word: %.4f' % 
                (timeSince(start_time), iter_id, 1-AR, WER, '|---Details:', self.delete_error, self.replace_error, self.insert_error, self.character_total, self.pred_character_total, self.word_total))
        else:
            print('iter_id: %6d, CER: %4f  WER: %4f \n %10s delete: %4d, replace: %4d, insert: %4d, gt_len: %4d, pred_len: %4d, total word: %.4f' % 
                (iter_id, 1-AR, WER, '|___Details:', self.delete_error, self.replace_error, self.insert_error, self.character_total, self.pred_character_total, self.word_total))
        if clear:
            self.clear()

    def forward(self, input, label, image, test=False, showing=False, iseval=False):
        self.showing = False
        self.softmax = input
        self.label = label
        self.image = image.permute(0,2,3,1).data.cpu().numpy()

    def decode_batch_2d(self):
        h, w, batch_size, dim = self.softmax.shape
        out_best_prob, out_best_idx = torch.max(F.softmax(self.softmax, -1), 3)
        out_best_idx = out_best_idx.data.cpu().numpy()
        out_best_prob = out_best_prob.data.cpu().numpy()
        return out_best_idx, out_best_prob 

    def parse(self):

        delete_total = 0
        replace_total = 0
        insert_total = 0
        len_total = 0
        correct_count = 0
        pre_total = 0
        word_total = 0
        all_total = 0        
        distance_wer_t = 0
        wordlen_t = 0
        idx_pred, prob_pred = self.decode_batch_2d()
        nH, nW, batch_size = self.softmax.shape[:-1]
        for i in range(batch_size):
            cur_out = idx_pred[:,:,i].copy()
            pred_idx = []
            pred_loc = []
            pointlist = []
            location_stack = [] # Stack-based eight-neighbor merge
            for x in range(0, nH):
                for y in range(0, nW):
                    if cur_out[x, y] > 0:
                        loc = []
                        location_stack.append([x, y])
                        idx_ = cur_out[x, y]
                        cur_out[x, y] = 0
                        pred_idx.append(idx_)
                        while len(location_stack):
                            location_ = location_stack.pop()
                            loc.append(location_)
                            x_, y_ = location_
                            if x_ and cur_out[x_-1, y_] == idx_:
                                location_stack.append([x_-1, y_])
                                cur_out[x_-1, y_] = 0
                            if y_ and cur_out[x_, y_-1] == idx_:
                                location_stack.append([x_, y_-1])
                                cur_out[x_, y_-1] = 0
                            if x_ < nH-1 and cur_out[x_+1, y_] == idx_:
                                location_stack.append([x_+1, y_])
                                cur_out[x_+1, y_] = 0
                            if y_ < nW-1 and cur_out[x_, y_+1] == idx_:
                                location_stack.append([x_, y_+1])
                                cur_out[x_, y_+1] = 0
                            if x_ < nH-1 and y_ < nW-1 and cur_out[x_+1, y_+1] == idx_:
                                location_stack.append([x_+1, y_+1])
                                cur_out[x_+1, y_+1] = 0
                            if x_ < nH-1 and y_ and cur_out[x_+1, y_-1] == idx_:
                                location_stack.append([x_+1, y_-1])
                                cur_out[x_+1, y_-1] = 0
                            if x_ and y_ and cur_out[x_-1, y_-1] == idx_:
                                location_stack.append([x_-1, y_-1])
                                cur_out[x_-1, y_-1] = 0
                            if x_ and y_ < nW-1 and cur_out[x_-1, y_+1] == idx_:
                                location_stack.append([x_-1, y_+1])
                                cur_out[x_-1, y_+1] = 0
                        mean_x = sum([loc_[0] for loc_ in loc])/len(loc)
                        mean_y = sum([loc_[1] for loc_ in loc])/len(loc)
                        pointlist.append([mean_x, mean_y]+[idx_])
                        pred_loc.append(loc)

            img_h = self.image.shape[1]
            img_w = self.image.shape[2]
            stride_h = int(img_h*2 / nH)
            stride_w = int(img_w*2 / nW)
           
            pointlistsort = sorted(pointlist, key = lambda x: x[1])
            startpoint = []
            totalline = []
            ynow = 0
            xnow = 0
            while pointlistsort != []:
                oneline = []
                ynow, xnow, _ = pointlistsort[0]
                oneline.append(pointlistsort[0])
                startpoint.append(pointlistsort[0])
                pointlistsort.remove(pointlistsort[0])
                yavg = ynow
                xavg = xnow
                for point in pointlistsort[:]:
                    if ((point[0] - ynow < 2 and point[0] - ynow > -2) or (point[0] - yavg < 2 and point[0] - yavg > -2)) and point[1] - xnow < 20:
                        oneline.append(point)
                        idxnow = point[2]
                        ynow, xnow, _ = point
                        yavg = 0.9 * yavg + 0.1 * ynow
                        pointlistsort.remove(point)
                totalline.append(oneline)

            startandtotal = zip(startpoint, totalline)
            startandtotalsort = sorted(startandtotal, key = lambda x: x[0][0])


            pre_list = []
            for item in startandtotalsort:
                for char in item[1]:
                    pre_list.append(char[2])
                pre_list.append(79)
            pre_list = pre_list[:-1]

            
            label_list = self.label[i][self.label[i]!=-1].tolist()
            label_list = [int(ele)+1 for ele in label_list]
            distance, (delete, replace, insert) = cal_distance(label_list, pre_list)
            distance_wer, wordlen = cal_distance_wer(label_list, pre_list, self.alphabet)
            wordlen_t += wordlen
            distance_wer_t += distance_wer
            delete_total += delete
            replace_total += replace
            insert_total += insert
            len_total += len(label_list)
            pre_total += len(pre_list)
            if distance == 0:
                word_total += 1 
            all_total += 1            

        self.delete_error += delete_total
        self.replace_error += replace_total
        self.insert_error += insert_total
        self.word_error += distance_wer_t
        self.pred_character_total += pre_total
        self.character_total += len_total
        self.word_total += wordlen_t
        self.images += all_total
        result = [delete_total, replace_total, insert_total, len_total ,correct_count, len_total, pre_total, word_total, all_total]
        return idx_pred, result, distance_wer_t, wordlen_t 








