# coding:utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime
#------------------------
from utils import *
from ExCTC import *
import cfgs as cfgs
import time
from tqdm import tqdm
import pdb
#------------------------
def display_cfgs(models):
    print('global_cfgs')
    cfgs.showcfgs(cfgs.global_cfgs)
    print('dataset_cfgs')
    cfgs.showcfgs(cfgs.dataset_cfgs)
    print('net_cfgs')
    cfgs.showcfgs(cfgs.net_cfgs)
    print('optimizer_cfgs')
    cfgs.showcfgs(cfgs.optimizer_cfgs)
    print('saving_cfgs')
    cfgs.showcfgs(cfgs.saving_cfgs)
    # for model in models:
    #     print(model)
#---------------------dataset
def load_dataset():
    train_loader = None

    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])

    val_data_set = cfgs.dataset_cfgs['dataset_val'](**cfgs.dataset_cfgs['dataset_val_args'])
    val_loader = DataLoader(val_data_set, **cfgs.dataset_cfgs['dataloader_val'])
    return (train_loader, test_loader, val_loader)
#---------------------network
def load_network():
    model = cfgs.net_cfgs['model'](**cfgs.net_cfgs['model_args'])

    if cfgs.net_cfgs['init_state_dict'] != None:
        model.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict']))    
    model.cuda()
    return (model)
#----------------------optimizer
def generate_optimizer(model):
    optimizer = cfgs.optimizer_cfgs['optimizer_0'](
                    model.parameters(),
                    **cfgs.optimizer_cfgs['optimizer_0_args'])
    optimizer_scheduler = cfgs.optimizer_cfgs['optimizer_0_scheduler'](
                    optimizer,
                    **cfgs.optimizer_cfgs['optimizer_0_scheduler_args'])
    return optimizer, optimizer_scheduler
#---------------------
def show_squeeze(data, attention_maps, which, ep, it):
    B, C, img_H, img_W = data.size()
    B_, C_, img_H_, img_W_ = attention_maps.size()
    font = cv2.FONT_HERSHEY_SIMPLEX

    image = (data[which][0].cpu().data.numpy()*127.5+127.5).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    attention_map = (attention_maps[which][0].cpu().data.numpy()*255).astype(np.uint8)
    attention_map = cv2.resize(attention_map, (img_W, img_H))
    attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_HOT)

    output = np.concatenate((image, attention_map), axis = 0)
    cv2.imwrite('SqueezeVis/image_{}_{}.jpg'.format(ep, it), output)

def show_fullpage(image, output, idx=0):
    img_h, img_w = image.size(2), image.size(3)
    output = torch.softmax(output, dim=-1)
    
    up_scale = 2
    # input image
    input_image = 255 * image[0,0].cpu().data.numpy()
    input_image = input_image.astype(np.uint8)
    input_image = cv2.resize(input_image, (up_scale*img_w,up_scale*img_h))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    # generate saliency map
    saliency_map = 255 * (1-output[:,:,:,0]).squeeze().cpu().data.numpy()
    saliency_map = saliency_map.astype(np.uint8)
    saliency_map = cv2.resize(saliency_map, (up_scale*img_w,up_scale*img_h))
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_HOT)
    # cv2.imwrite('saliency.jpg', saliency_map)
    # dense prediction map
    char_dict = cfgs.data_args['DICT']
    dense_pred = (np.ones((img_h*up_scale, img_w*up_scale))*255).astype(np.uint8)
    char_idx = output.topk(1)[1].squeeze()
    x_, y_ = int(img_w*up_scale/output.size(1)), int(img_h*up_scale/output.size(0))
    for x in range(0, char_idx.size(1)):
        for y in range(0, char_idx.size(0)):
            if int(char_idx[y,x])>0:
                cv2.putText(dense_pred, char_dict[int(char_idx[y,x])-1], ((x+1)*x_, (y+1)*y_), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0), 1)
    dense_pred = cv2.cvtColor(dense_pred, cv2.COLOR_GRAY2RGB)
    # 
    show_img = np.concatenate([input_image, saliency_map, dense_pred], axis=0)
    cv2.imwrite('FullpageVis/test_{0:04d}_img_saliency_densepred.jpg'.format(idx), show_img)

    # pdb.set_trace()

#---------------------testing stage
def test(test_loader, val_loader, model, tools):
    model.eval()
    import datetime
    # start_time = datetime.datetime.now()
    parser = Sequence(cfgs.data_args['DICT'])
    val_parser = CTC_AR_counter()

    loss_aver = 0
    total_result = [0]*9
    diswert = 0
    wlent = 0
    # using line-level images as validation set
    for it, (sample_batched) in enumerate(val_loader):
        inputs = sample_batched['image']
        labels = sample_batched['label'].squeeze(1)

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # output = model(inputs, labels, iseval=True)
        output, attnmap, ctc_loss, aceloss, concentrate_loss = model(inputs, labels)
        val_parser.add_iter(output, labels)
    val_parser.show()
    start_time = time.time()
    # using full-page images as test set
    # for it, (sample_batched) in enumerate(tqdm(test_loader)):
    for it, (sample_batched) in enumerate(test_loader):
        inputs = sample_batched['image']
        labels = sample_batched['label'].squeeze(1)

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        output = model(inputs, labels, temperature=1, IFA_inference=True)
        parser(output, labels, inputs, iseval=True)
        prediction, result, diswer, wlen = parser.parse()
        
        for i, ele in enumerate(result):
            total_result[i] += ele 
        diswert += diswer
        wlent += wlen
        show_fullpage(inputs, output, it)
        # if it % 50 == 0:
        #     parser.show(it, False, start_time)
    parser.show(it, True, start_time)
#---------------------

#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    # prepare nets, optimizers and data
    model = load_network()
    # pdb.set_trace()
    if cfgs.global_cfgs['state'] == 'Train':
        display_cfgs(model)
    else:
        print(cfgs.net_cfgs)
    optimizer, optimizer_scheduler = generate_optimizer(model)
    loss_func = CTC()
    train_loader, test_loader, val_loader = load_dataset()
    print('preparing done')
    # --------------------------------
    # prepare tools
    loss_counter_CTC = Loss_counter()
    loss_counter_WHACE = Loss_counter()
    loss_counter_Concentrate = Loss_counter()
    #---------------------------------
    if cfgs.global_cfgs['state'] == 'Test':
        test((test_loader),
              val_loader, 
             model, 
            [])
        exit()
    # --------------------------------
    total_iters = len(train_loader)
    for nEpoch in range(0, cfgs.global_cfgs['epoch']):
        for batch_idx, sample_batched in enumerate(train_loader):
            model.train()
            # data prepare
            inputs = sample_batched['image']
            labels = sample_batched['label'].squeeze(1)
            inputs = inputs.cuda()
            labels = labels.cuda()

            temperature = 0.1 if nEpoch > 10 else 1

            output, attnmap, ctc_loss, aceloss, concentrate_loss = model(inputs, labels, temperature)

            loss = ctc_loss
            if nEpoch > 5:
                loss = loss + aceloss
            if nEpoch > 10:
                loss = loss + concentrate_loss

            loss_counter_CTC.add_iter(ctc_loss)
            loss_counter_WHACE.add_iter(aceloss)
            loss_counter_Concentrate.add_iter(concentrate_loss)
            # update network
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # visualization and saving
            if batch_idx % cfgs.global_cfgs['show_interval'] == 0 and batch_idx != 0:
                show_squeeze(inputs, attnmap, 0, nEpoch, batch_idx)
                print(datetime.datetime.now().strftime('%H:%M:%S'))
                print('Epoch: {}, Iter: {}/{}, Loss CTC: {}, Loss WHACE: {}, Loss Concentrate: {}'.format(
                                    nEpoch,
                                    batch_idx,
                                    total_iters,
                                    loss_counter_CTC.get_loss(),
                                    loss_counter_WHACE.get_loss(),
                                    loss_counter_Concentrate.get_loss()))
            if batch_idx % cfgs.global_cfgs['test_interval'] == 0 and batch_idx != 0:
                test((test_loader),
                     val_loader, 
                     model, 
                    [])
            if nEpoch % cfgs.saving_cfgs['saving_epoch_interval'] == 0 and \
                batch_idx % cfgs.saving_cfgs['saving_iter_interval'] == 0 and \
                batch_idx != 0:
                torch.save(model.state_dict(),
                         cfgs.saving_cfgs['saving_path'] + 'E{}_I{}-{}.pth'.format(
                            nEpoch, batch_idx, total_iters))   
        optimizer_scheduler.step()
