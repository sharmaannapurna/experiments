import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import CTCLoss
import torch.nn.functional as F
from hw import hw_dataset
from hw import cnn_lstm
from hw.hw_dataset import HwDataset

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

import numpy as np
import cv2
import sys
import json
import os
from utils import string_utils, error_rates
import time
import random
import yaml

from utils.dataset_parse import load_file_list

with open(sys.argv[1]) as f:
    config = yaml.load(f)

hw_network_config = config['network']['hw']
pretrain_config = config['pretraining']

char_set_path = hw_network_config['char_set_path']

with open(char_set_path) as f:
    char_set = json.load(f)

idx_to_char = {}
for k,v in char_set['idx_to_char'].items():
    idx_to_char[int(k)] = v

training_set_list = load_file_list(pretrain_config['training_set'])
train_dataset = HwDataset(training_set_list,
                          char_set['char_to_idx'], augmentation=True,
                          img_height=hw_network_config['input_height'])

train_dataloader = DataLoader(train_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=True, num_workers=0, drop_last=True,
                             collate_fn=hw_dataset.collate)

batches_per_epoch = int(pretrain_config['hw']['images_per_epoch']/pretrain_config['hw']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = HwDataset(test_set_list,
                         char_set['char_to_idx'],
                         img_height=hw_network_config['input_height'])

test_dataloader = DataLoader(test_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=False, num_workers=0,
                             collate_fn=hw_dataset.collate)



criterion = CTCLoss(blank=0, reduction='sum', zero_infinity=True)

hw = cnn_lstm.create_model(hw_network_config)
hw.cuda()


optimizer = torch.optim.Adam(hw.parameters(), pretrain_config['hw']['learning_rate'])
dtype = torch.cuda.FloatTensor

lowest_loss = np.inf
cnt_since_last_improvement = 0
for epoch in range(1000):
    print("Epoch", epoch)
    sum_loss1 = 0.0
    total_loss =0.0
    steps1 = 0.0
    step2=0.0
    hw.train()
    for x in train_dataloader:
        optimizer.zero_grad()
        with torch.no_grad():
            line_imgs = x['line_imgs'].type(dtype)#, requires_grad=False)
            labels =  x['labels']#, requires_grad=False)
            label_lengths = x['label_lengths']#, requires_grad=False)

        preds = hw(line_imgs).cpu()
        
        # print(preds.shape)
        output_batch = preds.permute(1,0,2)
        # output_batch=preds
        # print(output_batch.shape)
        out = output_batch.cpu().detach().numpy()
        # out.requires_grad(True)

        # out=F.softmax(out, dim=2).
        batch_size = preds.size(1)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        # print('string sizes:',preds_size, "\n", label_lengths)

        # print "before"
        loss = criterion(preds, labels, preds_size, label_lengths)
        total_loss+=loss
        # print(loss)# print "after"

        step2+=1
        loss.backward()
        optimizer.step()
        for i, gt_line in enumerate(x['gt']):
            logits = out[i,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            # print(labels, pred,"\n", raw_pred, "\n", pred_str, "\n",gt_line)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss1 += cer
            steps1 += 1

        # print("out:", out.shape, out)
        # print("logits:", logits.shape)  

    print("Train Loss", sum_loss1/steps1, total_loss/step2)
    print("Real Epoch", train_dataloader.epoch)

    sum_loss = 0.0
    steps = 0.0
    hw.eval()

    for x in test_dataloader:
        with torch.no_grad():
            line_imgs = x['line_imgs'].type(dtype)#, requires_grad=False, volatile=True)
            labels =  x['labels']#, requires_grad=False, volatile=True)
            label_lengths = x['label_lengths']#, requires_grad=False, volatile=True)

        preds = hw(line_imgs).cpu()

        output_batch = preds.permute(1,0,2)
        out = output_batch.cpu().detach().numpy()
        batch_size = preds.size(1)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        # print('string sizes:',preds_size, "\n", label_lengths)

        # print "before"
        loss = criterion(preds, labels, preds_size, label_lengths)
        total_loss+=loss


        for i, gt_line in enumerate(x['gt']):
            logits = out[i,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss/steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss/steps
        print("Saving Best")

        if not os.path.exists(pretrain_config['snapshot_path']):
            os.makedirs(pretrain_config['snapshot_path'])

        torch.save(hw.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'hw.pt'))

    print("Test Loss", sum_loss/steps, lowest_loss)
    print("")

    if cnt_since_last_improvement >= pretrain_config['hw']['stop_after_no_improvement'] and lowest_loss<0.9:
        break
