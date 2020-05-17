#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:33:53 2020

@author: milly
"""

import time
import numpy as np
import os
import torch
from torch import nn
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, Dataset
import sys
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict

class Generator(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1):
        super(Generator, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features1=64
        self.features2=self.features1*2 #128
        self.features3=self.features2*2 #256
        self.features4=self.features3*3 #512
        
        self.encoder1 = self._enc_block(self.in_channels, 
                                        self.features1, "enc1")
        self.encoder2 = self._enc_block(self.features1, 
                                        self.features2, "enc2")
        self.encoder3 = self._enc_block(self.features2, 
                                        self.features3, "enc3")
        self.encoder4 = self._enc_block(self.features3, 
                                        self.features4, "enc4")
        self.encoder_final = self._enc_block(self.features4,
                                             self.features4, "enc_final")
        
        self.decoder_m = self._dec_block(self.features4, 
                                         self.features4, "decm")
        
        self.decoder_start = self._dec_block(self.features4*2, 
                                             self.features4, "dec_start") 
        self.decoder4 = self._dec_block(self.features4*2, 
                                        self.features3, "dec4")
        self.decoder3 = self._dec_block(self.features3*2, 
                                        self.features2, "dec3")
        self.decoder2 = self._dec_block(self.features2*2, 
                                        self.features1, "dec2")
        self.decoder1 = self._dec_block(self.features1*2, 
                                        self.out_channels, "dec1")
        
    def _enc_block(self, in_channels, features, name):
        return nn.Sequential(OrderedDict(
            [
                (name + "conv", nn.Conv2d(in_channels, 
                                          features,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)),
                (name + "norm", nn.BatchNorm2d(num_features=features)),
                (name + "LReLU", nn.LeakyReLU(0.2))
                ]
            ))
    
    def _dec_block(self, in_channels, features, name):
        return nn.Sequential(OrderedDict(
            [
                (name + "tconv", nn.ConvTranspose2d(in_channels,
                                                    features,
                                                    kernel_size=4,
                                                    stride=2,
                                                    padding=1)),
                (name + "norm", nn.BatchNorm2d(num_features=features)),
                (name + "LReLU", nn.LeakyReLU(0.2))
                ]
            ))
    
    def forward(self, x):
        
        #x = inx512x512
        x1 = self.encoder1(x) # 64x256x256
        x2 = self.encoder2(x1) # 128x128x128
        x3 = self.encoder3(x2) # 256x64x64
        x4 = self.encoder4(x3) # 512x32x32
        x5 = self.encoder_final(x4) # 512x16x16
        x6 = self.encoder_final(x5) # 512x8x8
        x7 = self.encoder_final(x6) # 512x4x4
        x8 = self.encoder_final(x7) # 512x2x2
        
        m = self.encoder_final(x8) # 512x1x1
        y = self.decoder_m(m) # 512x2x2
        
        y = torch.cat((y, x8), dim=1) # 1024x2x2
        y = self.decoder_start(y) # 512x4x4
        y = torch.cat((y, x7), dim=1) # 1024x4x4
        y = self.decoder_start(y) # 512x8x8
        y = torch.cat((y, x6), dim=1) # 1024x8x8
        y = self.decoder_start(y) # 512x16x16
        y = torch.cat((y, x5), dim=1) # 1024x16x16
        y = self.decoder_start(y)  # 512x32x32
        y = torch.cat((y, x4), dim=1) # 1024x32x32
        y = self.decoder4(y) # 256x64x64
        y = torch.cat((y, x3), dim=1) # 512x64x64
        y = self.decoder3(y) # 128x128x128
        y = torch.cat((y, x2), dim=1) # 256x128x128
        y = self.decoder2(y) # 64x256x256
        y = torch.cat((y, x1), dim=1) # 128x256x256
        y = self.decoder1(y) # outx512x512
        return y
    
 
class Discriminator(nn.Module):
    
    def __init__(self, in_channels=2, out_channels=1):
        super(Discriminator, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features1=64
        self.features2=self.features1*2 #128
        self.features3=self.features2*2 #256
        self.features4=self.features3*2 #512
        
        self.dis1 = self._block(self.in_channels, 
                                self.features1, "dis1")
        self.dis2 = self._block(self.features1, 
                                self.features2, "dis2")
        self.dis3 = self._block(self.features2, 
                                self.features3, "dis3")
        self.dis4 = self._block(self.features3,
                                self.features4, stride=1, name="dis4")
        self.conv = nn.Conv2d(self.features4, 
                              self.out_channels, 
                              kernel_size=4, stride=1, padding=2)
        
    def _block(self, in_channels, features, name,
               kernel_size=4, stride=2, padding=1):
        
        return nn.Sequential(OrderedDict(
            [
                (name + "conv", nn.Conv2d(in_channels, 
                                          features,
                                          kernel_size,
                                          stride,
                                          padding)),
               # (name + "norm", nn.BatchNorm2d(num_features=features)),
                (name + "ReLU", nn.ReLU())
                ]
            ))
    
    
    def forward(self, x):
        x = self.dis1(x)
        x = self.dis2(x)
        x = self.dis3(x)
        x = self.dis4(x)
        x = self.conv(x)
        x = x.view(-1, 64*64)
        return x
    
class NumpyArraySet(Dataset):
    
    def __init__(self, load_folder, name, transforms=None):
        
        self.list_dir = os.listdir(load_folder)
        self.load_folder = load_folder
        self.name = name + "_{}.npy"
        self.transforms = transforms
        
    def __len__(self):
        return(len(self.list_dir))
    
    def __getitem__(self, index):
        
        path = os.path.join(self.load_folder, self.name.format(index))
        sample = torch.tensor(np.load(path)).reshape((1,512,512))
        if self.transforms:
            sample = self.transforms(sample)
        return sample

class Normalize():
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, image):
        image_max = image.max().item()
        image_min = image.min().item()
        ratio = (self.max_val - self.min_val)/(image_max - image_min)
        image = image*ratio + self.min_val - image_min/(image_max - image_min)
        return image
    
def max_divide(image):
    return image/image.max()

class GaussianNoise():
    def __init__(self, percent):
        self.percent = percent/100
        
    def __call__(self, image):
        noise = torch.randn_like(image)
        noise *= self.percent * image.norm() / noise.norm()
        image += noise
        return image
        
class RandomMask():
    def __init__(self, min_missing_angles_deg, max_missing_angles_deg):
        self.max_angles = max_missing_angles_deg
        self.min_angles = min_missing_angles_deg
        
    def __call__(self, sino):
        max_val = int(len(sino[0])*self.max_angles/180)
        min_val = int(len(sino[0])*self.min_angles/180)
        
        if min_val == max_val:
            missing = min_val
        else:
            missing = np.random.randint(min_val, max_val)
        
        start = int(len(sino[0])/2-missing/2)
        end = int(len(sino[0])/2+missing/2)
        lim = copy.deepcopy(sino)
        lim[:,start:end,:] = torch.zeros_like(lim[:,start:end,:])
        
        mask = torch.ones_like(sino)
        mask[:,start:end,:] = torch.zeros_like(mask[:,start:end,:])

        return {"full" : sino, "lim" : lim, "mask" : mask}
    
class ContextWeight():
    def __init__(self, function, example_specific = False):
        
        self.function = function
        self.example_specific = example_specific
  
    def __call__(self, img, mask):
        image = img*1 #so not inplace
        if self.function:
            weights = mask[0].transpose(0,1)[0].reshape(512)
            top = torch.nonzero(weights[:256])#.reshape(256)
            bottom = torch.nonzero(weights[256:]) +256
            weights = torch.where(weights == 0, 
                                  torch.ones_like(weights), 
                                  torch.zeros_like(weights))
            if self.example_specific:
                x = torch.linspace(-1, 1, len(weights)).to(device)
                weights[top] = self.function(x[top], x[top][-1])
                weights[bottom] = self.function(x[bottom], x[bottom][0])
            else:
                weights[top] = self.function(
                    torch.linspace(-1, 1, len(weights))[top]).to(device)
                weights[bottom] = self.function(
                    torch.linspace(-1, 1, len(weights))[bottom]).to(device)
    
            image[0] = (weights*image[0].transpose(0,1)).transpose(0,1)
        return image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

class ContextWindow:   
    def __init__(self, n_extra_rows, image_height):
        self.name = "window_" + str(int(n_extra_rows))
        self.n_extra_rows = int(n_extra_rows)
        self.image_height = int(image_height)
        if self.n_extra_rows != 0:
            self.fractional_extra = self.n_extra_rows /self.image_height / 2
        else:
            self.fractional_extra = 0    
    def __call__(self, tensor1D, start):
        return torch.where(abs(tensor1D) < abs(start) + self.fractional_extra, 
                           torch.ones_like(tensor1D), 
                           torch.zeros_like(tensor1D))
    
class ContextCos:
    def __init__(self, power):
        self.name = "cos_" + str(power)
        self.n = power
    def __call__(self, tensor1D, start):
        return ((torch.cos(np.pi*(tensor1D - start)) + 1)/2) ** self.n
    
def train_gan(generator, discriminator, 
              g_optimizer, d_optimizer,
              adv_loss, pixel_loss,
              data_loader, gamma,
              g_context_function=None, g_example_specific_context=False,
              d_context_function=None, d_example_specific_context=False,
              g_training=True, d_training=True):
    
    data_len = 0 #counting data to average loss
    
    #to save average epoch loss
    g_epoch_loss = 0
    g_epoch_pixel_loss = 0
    g_epoch_adv_loss = 0
    d_epoch_loss = 0
    d_epoch_fake_loss = 0
    d_epoch_real_loss = 0
    
    #to save average batch loss
    g_batch_loss = np.zeros(len(data_loader))
    g_batch_pixel_loss = np.zeros(len(data_loader))
    g_batch_adv_loss = np.zeros(len(data_loader))
    d_batch_loss = np.zeros(len(data_loader))
    d_batch_fake_loss = np.zeros(len(data_loader))
    d_batch_real_loss = np.zeros(len(data_loader))
    
    g_context_weight = ContextWeight(g_context_function, 
                                     g_example_specific_context)
    d_context_weight = ContextWeight(d_context_function, 
                                     d_example_specific_context)
        
    for i, sample in enumerate(data_loader):

        for name in sample:
            sample[name] = sample[name].to(device)
            
        bs = len(sample["lim"])
            
        real_label = torch.full((bs, 64*64), 1, dtype=torch.float).to(device)
        fake_label = torch.full((bs, 64*64), 0, dtype=torch.float).to(device)
            
        generator.train(g_training)
        discriminator.train(d_training)
        
        x = sample["lim"]

        g_out = generator(x)
        
        #-------------------discriminator train---------------------------
        discriminator.zero_grad()
        
        #----real batch-------
        d_real_in = torch.cat(
            (torch.cat(list(map(d_context_weight, 
                                sample["full"], 
                                sample["mask"]))), 
             sample["lim"]), dim=1)
        d_real_out = discriminator(d_real_in)
        d_real_loss = adv_loss(d_real_out, real_label)
        
        #----fake batch-------
        d_fake_in = torch.cat(
                    (torch.cat(list(map(d_context_weight, 
                                        copy.deepcopy(g_out.detach()), 
                                        sample["mask"]))), 
                    sample["lim"]), dim=1)
        d_fake_out = discriminator(d_fake_in)
        d_fake_loss = adv_loss(d_fake_out, fake_label)
        
        d_loss = 0.5 * d_real_loss + 0.5 * d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        #---------------------generator train-----------------------------
        generator.zero_grad()
        #discriminator.zero_grad() - not necessary

        g_out_weighted = torch.cat(list(map(g_context_weight, 
                                            g_out, 
                                            sample["mask"])))
        g_target_weighted = torch.cat(list(map(g_context_weight, 
                                               sample["full"], 
                                               sample["mask"])))
        g_pixel_loss = pixel_loss(g_out_weighted, g_target_weighted)
        
        d_fake_in = torch.cat(
            (torch.cat(list(map(d_context_weight,
                                g_out, #not detached
                                sample["mask"]))),
             sample["lim"]), dim=1)
        d_fake_out_new = discriminator(d_fake_in)
        g_adv_loss = adv_loss(d_fake_out_new, real_label)
        
        g_loss = g_adv_loss + gamma * g_pixel_loss
        g_loss.backward()
        g_optimizer.step()
        
        #-------------------saving loss values----------------------------
        
        #multiply by batch size because assuming loss setting is mean
        g_epoch_loss += g_loss.item() * bs
        g_epoch_pixel_loss += g_pixel_loss.item() * bs
        g_epoch_adv_loss += g_adv_loss.item() * bs
        
        d_epoch_loss += d_loss.item() * bs
        d_epoch_fake_loss += d_fake_loss.item() * bs
        d_epoch_real_loss += d_real_loss.item() * bs
        
        g_batch_loss[i] = g_loss.item()
        g_batch_pixel_loss[i] = g_pixel_loss.item()
        g_batch_adv_loss[i] = g_adv_loss.item()
        
        d_batch_loss[i] = d_loss.item()
        d_batch_fake_loss[i] = d_fake_loss.item()
        d_batch_real_loss[i] = d_real_loss.item()
        
        data_len += bs
    
    #now divide by total data
    g_epoch_loss /= data_len
    g_epoch_pixel_loss /= data_len
    g_epoch_adv_loss /= data_len
    
    d_epoch_loss /= data_len
    d_epoch_fake_loss /= data_len
    d_epoch_real_loss /= data_len
    
    epoch_data = {"generator" : generator,
                  "discriminator" : discriminator,
                  "g_epoch_loss" : g_epoch_loss,
                  "g_epoch_pixel_loss" : g_epoch_pixel_loss,
                  "g_epoch_adv_loss" : g_epoch_adv_loss,
                  "d_epoch_loss" : d_epoch_loss,
                  "d_epoch_fake_loss" : d_epoch_fake_loss,
                  "d_epoch_real_loss" : d_epoch_real_loss,
                  "g_batch_loss" : g_batch_loss,
                  "g_batch_pixel_loss" : g_batch_pixel_loss,
                  "g_batch_adv_loss" : g_batch_adv_loss,
                  "d_batch_loss" : d_batch_loss,
                  "d_batch_fake_loss" : d_batch_fake_loss,
                  "d_batch_real_loss" : d_batch_real_loss}
                          
    return epoch_data

#script automatically makes folders to save data 
#and text file to record hyperparameters
    
if torch.cuda.is_available():
    device = torch.device('cuda:1') 
else:
    device = 'cpu'

root_dir = os.getcwd()

stats_folder = os.path.join(root_dir, "epoch_stats")
state_dict_folder = os.path.join(root_dir, "state_dicts")
epoch_output_folder = os.path.join(root_dir, "epoch_outputs")
data_file_path = os.path.join(root_dir, "hyperparam_track.txt")

if os.path.isdir(stats_folder) == False:
    os.mkdir(stats_folder)
if os.path.isdir(state_dict_folder) == False:
    os.mkdir(state_dict_folder)
if os.path.isdir(epoch_output_folder) == False:
    os.mkdir(epoch_output_folder)
    

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#=========================user chosen hyperparameters=========================

dataset_folder = "/media/newhd/milly/SEM2"
dataset_name = "sinograms"
trainset = '2'
save_example = 10

loading_checkpoint =False
checkpoint_name = "200_300_2_13"
    
epochs = 400
bs = 8

schedule_gamma = False
gamma_start = 120
gamma_end = 120

#optimizer (ADAM) parameters for generator and discriminator
g_lr = 0.002; b1_g = 0.5; b2_g = 0.999 
d_lr = 0.002; b1_d = 0.5; b2_d = 0.999

schedule_g_lr = True
g_lr_reduce_ratio = 0.0025
schedule_d_lr = True
d_lr_reduce_ratio = 0.0025

g_context_choice = "none"
d_context_choice = "none"

angle_min = 90
angle_max = 90

gaussian_noise_percent = 5

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#=============================================================================

        
hyperparams = {"g context" : g_context_choice,
               "d context" : d_context_choice,
               "angle min" : angle_min,
               "angle max" : angle_max,
               "schedule gamma" : schedule_gamma,
               "gamma start" : gamma_start,
               "gamma end" : gamma_end,
               "gaussian noise" : gaussian_noise_percent,
               "schedule_g_lr" : schedule_g_lr, 
               "g_lr_reduce_ratio" : g_lr_reduce_ratio,
               "schedule_d_lr" : schedule_d_lr,
               "d_lr_reduce_ratio" : d_lr_reduce_ratio,
               "g_lr" : g_lr, "b1_g" : b1_g, "b2_g" : b2_g,
               "d_lr" : d_lr, "b1_d" : b1_d, "b2_d" : b2_d,
               "bs" : bs,
               }

if not os.path.isfile(data_file_path):
    data_file = open(data_file_path, 'w')
    data_file.write("0, run, name, epoch start, epoch end")
    data_file.write(", ".join(hyperparams))
    data_file.write("\n")
    data_file.close()

run=1
epoch_start = 0


#--------------loading checkpoint parameters from textfile--------------------
without_checkpoint = ""
checkpoint_found = False
if loading_checkpoint:
    
    data_file = open(data_file_path, 'r')
    data_list = data_file.readlines()
    data_file.close()
    
    for line in data_list:
        if line != "\n":
            line_list = line.split(", ")
            if line_list[1] == checkpoint_name:
                checkpoint_found = True
                run = int(line_list[0])
                epoch_start = int(line_list[3])
                prev_epochs = epoch_start - int(line_list[2])
                trainset = checkpoint_name.split("_")[2]
                for i, name in enumerate(hyperparams):
                    hyperparams[name] = line_list[4 + i]
                    if hyperparams[name] == "True":
                        hyperparams[name] = True
                    if hyperparams[name] == "False":
                        hyperparams[name] = False
                    
    if not checkpoint_found:
        while without_checkpoint not in ["n", "y"]:
            message = "Checkpoint not found, continue as new run? [y/n]: "
            without_checkpoint = input(message)
        if without_checkpoint == "n":
            sys.exit(1)     
    
    old_gamma_end = float(hyperparams["gamma end"])
    old_gamma_start = float(hyperparams["gamma start"])
    old_gamma_end -= (old_gamma_start - old_gamma_end) * epochs / prev_epochs
    gamma_end =  old_gamma_end
    if gamma_end < 0:
        gamma_end = 0
    hyperparams["gamma start"] = hyperparams["gamma end"]
    hyperparams["gamma end"] = gamma_end
    

#------------------implementing hyperparameter choices-------------------------
 
#---------context functions---------
g_context=None
d_context=None
for context, choice in [(g_context, hyperparams["g context"]), 
                        (d_context, hyperparams["d context"])]:
    if choice.split("_")[0] == "cos":
        context = ContextCos(float(choice.split("_")[1]))
    elif choice.split("_")[0] == "window":
        #change so this is so image height (512) a chosen parameter ?
        context = ContextWindow(int(choice.split("_")[1]), 512) 
    elif choice == "none" or choice =='None':
        context = None
    else:
        err = "Context choice \"{}\" invalid, continuing with None.".format(choice)
        print(err)


#-----Loading sinograms from trainset choice-----
trainset_list = list(filter(lambda x: dataset_name in x, 
                            os.listdir(dataset_folder)))
trainset_num_list = list(map(lambda x: x.split("_")[-1],
                             trainset_list))

trainset_name = trainset_list[trainset_num_list.index(trainset)]

sino_folder = "/media/newhd/milly/SEM2/{}/".format(trainset_name)

#normalize after noise applied, what would happen with real data
transforms = transforms.Compose([
    GaussianNoise(float(hyperparams["gaussian noise"])),
    #Normalize(0, 1),
    lambda image: image/image.max().item(),
    RandomMask(float(hyperparams["angle min"]), 
               float(hyperparams["angle max"]))
    ])

data = NumpyArraySet(sino_folder, "sino", transforms=transforms)
dataloader = DataLoader(data, batch_size=int(hyperparams["bs"]), shuffle=True)

#---------initialising networks---------
generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)
 
#--------initialising optimisers--------           
g_optimizer = optim.Adam(generator.parameters(), 
                         lr=float(hyperparams["g_lr"]), 
                         betas=(float(hyperparams["b1_g"]),
                                float(hyperparams["b2_g"])))
d_optimizer = optim.Adam(discriminator.parameters(), 
                         lr=float(hyperparams["d_lr"]), 
                         betas=(float(hyperparams["b1_d"]),
                                float(hyperparams["b2_d"])))

if hyperparams["schedule_g_lr"]:
    g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        g_optimizer, 
        lambda epoch: (1-float(hyperparams["g_lr_reduce_ratio"])*epoch),
        last_epoch=-1)
else: 
    g_lr_scheduler = None

if hyperparams["schedule_d_lr"]:
    d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        d_optimizer, 
        lambda epoch: (1-float(hyperparams["d_lr_reduce_ratio"])*epoch),
        last_epoch=-1)
else: 
    d_lr_scheduler = None

#loading optimizer and network previous states
if loading_checkpoint and without_checkpoint == "":
    print("Loading checkpoints")
    state_dicts = torch.load(os.path.join(state_dict_folder, checkpoint_name),
                             map_location=device)
    generator.load_state_dict(state_dicts["g_state"])
    discriminator.load_state_dict(state_dicts["d_state"])
    g_optimizer.load_state_dict(state_dicts["g_optimizer_state"])
    d_optimizer.load_state_dict(state_dicts["d_optimizer_state"])
    if hyperparams["schedule_g_lr"]:
        g_lr_scheduler.load_state_dict(state_dicts["g_lr_scheduler"])
    if hyperparams["schedule_d_lr"]:
        d_lr_scheduler.load_state_dict(state_dicts["d_lr_scheduler"])    

#-----loss functions-----
adv_loss = nn.BCEWithLogitsLoss()
gen_loss = nn.L1Loss()

#------for saving training stats---------
batches = len(dataloader)

g_epoch_loss = np.zeros(epochs)
g_epoch_pixel_loss = np.zeros(epochs)
g_epoch_adv_loss = np.zeros(epochs)
d_epoch_loss = np.zeros(epochs)
d_epoch_fake_loss = np.zeros(epochs)
d_epoch_real_loss = np.zeros(epochs)

g_batch_loss = np.zeros(epochs * batches)
g_batch_pixel_loss = np.zeros(epochs * batches)
g_batch_adv_loss = np.zeros(epochs * batches)
d_batch_loss = np.zeros(epochs * batches)
d_batch_fake_loss = np.zeros(epochs * batches)
d_batch_real_loss = np.zeros(epochs * batches)

g_epoch_outputs = np.zeros((epochs, 512, 512))
d_epoch_real_outputs = np.zeros((epochs, 64, 64))
d_epoch_fake_outputs = np.zeros((epochs, 64, 64))

#----------start of training---------
train_start = time.time()
gamma = float(hyperparams["gamma start"])
for epoch in range(epoch_start, epoch_start+epochs):
    
    epoch_start_time = time.time()
    
    print('-' * 5)
    print('Epoch: {}'.format(epoch + 1))
    
    if bool(hyperparams["schedule gamma"]) == True and gamma != 0:
        gamma -= (float(hyperparams["gamma start"])
                  -float(hyperparams["gamma end"]))/epochs
    if gamma < 0:
        gamma = 0
    print("Gamma: {}".format(gamma))
    if hyperparams["schedule_g_lr"]:
        print("G lr: {}".format(g_lr_scheduler.get_lr()))
    if hyperparams["schedule_d_lr"]:
        print("D lr: {}".format(d_lr_scheduler.get_lr()))
    
    result = train_gan(generator, discriminator, 
                       g_optimizer, d_optimizer,
                       adv_loss, gen_loss, 
                       dataloader, gamma,
                       g_context, True,
                       d_context, True)
    
    discriminator = result["discriminator"]
    generator = result["generator"]
    
    if d_lr_scheduler:
            d_lr_scheduler.step()
    if g_lr_scheduler:
            g_lr_scheduler.step()
    
    #-------------saving stats-------------
    index = epoch-epoch_start #counts 0,1,2...
    batch_start = batches*index
    batch_end = batches*(index+1)    
    
    g_epoch_loss[index] = result["g_epoch_loss"]
    g_epoch_pixel_loss[index] = result["g_epoch_pixel_loss"]
    g_epoch_adv_loss[index] = result["g_epoch_adv_loss"]
    d_epoch_loss[index] = result["d_epoch_loss"]
    d_epoch_fake_loss[index] = result["d_epoch_fake_loss"]
    d_epoch_real_loss[index] = result["d_epoch_real_loss"]
    
    g_batch_loss[batch_start:batch_end] = result["g_batch_loss"]
    g_batch_pixel_loss[batch_start:batch_end] = result["g_batch_pixel_loss"]
    g_batch_adv_loss[batch_start:batch_end] = result["g_batch_adv_loss"]
    d_batch_loss[batch_start:batch_end] = result["d_batch_loss"]
    d_batch_fake_loss[batch_start:batch_end] = result["d_batch_fake_loss"]
    d_batch_real_loss[batch_start:batch_end] = result["d_batch_real_loss"]
    
    print("Generator total loss: {}".format(result["g_epoch_loss"]))
    print("Generator pixel loss: {}".format(result["g_epoch_pixel_loss"]))
    print("Generator adversarial loss: {}".format(result["g_epoch_adv_loss"]))
    print("Discriminator total loss: {}".format(result["d_epoch_loss"]))
    print("Discriminator fake loss: {}".format(result["d_epoch_fake_loss"]))
    print("Discriminator real loss: {}".format(result["d_epoch_real_loss"]))

    epoch_end_time = time.time()
    print("Epoch time: {:.2f}".format(epoch_end_time-epoch_start_time))
    
    #--------saving output examples--------
    with torch.set_grad_enabled(False):
        sample = data[save_example]
        z = sample["lim"].reshape(1,1,512,512).to(device)
        generator.eval()
        gen_sino = generator(z.to(device))
    
        discriminator.eval()
        d_context_weight = ContextWeight(d_context, True)
        d_real_in = torch.cat(
            (torch.cat(list(map(d_context_weight, 
                                sample["full"].reshape(1,1,512,512).to(device), 
                                sample["mask"].reshape(1,1,512,512).to(device)
                                ))), 
            sample["lim"].reshape(1,1,512,512).to(device)), dim=1)
        d_real_out = discriminator(d_real_in)
        
        d_fake_in = torch.cat(
            (torch.cat(list(map(d_context_weight, 
                                gen_sino.reshape(1,1,512,512).to(device), 
                                sample["mask"].reshape(1,1,512,512).to(device)
                                ))), 
            sample["lim"].reshape(1,1,512,512).to(device)), dim=1)
        d_fake_out = discriminator(d_fake_in)
        
        g_epoch_outputs[index] = gen_sino.cpu().numpy().reshape(512,512)
        d_epoch_real_outputs[index] = d_real_out.cpu().numpy().reshape(64,64)
        d_epoch_fake_outputs[index] = d_fake_out.cpu().numpy().reshape(64,64)

train_end = time.time()

print("Total time for {} epochs: {:.2f}".format(epochs, train_end-train_start))

stats_to_save = {"g_epoch_loss" : g_epoch_loss,
                 "g_epoch_pixel_loss" : g_epoch_pixel_loss,
                 "g_epoch_adv_loss" : g_epoch_adv_loss,
                 "d_epoch_loss" : d_epoch_loss,
                 "d_epoch_fake_loss" : d_epoch_fake_loss,
                 "d_epoch_real_loss" : d_epoch_real_loss,
                 "g_batch_loss" : g_batch_loss,
                 "g_batch_pixel_loss" : g_batch_pixel_loss,
                 "g_batch_adv_loss" : g_batch_adv_loss,
                 "d_batch_loss" : d_batch_loss,
                 "d_batch_fake_loss" : d_batch_fake_loss,
                 "d_batch_real_loss" : d_batch_real_loss}

epoch_outputs = {"g_epoch_outputs" : g_epoch_outputs,
                 "d_epoch_real_outputs" : d_epoch_real_outputs,
                 "d_epoch_fake_outputs" : d_epoch_fake_outputs}

if g_lr_scheduler:
    g_lr_scheduler_state = copy.deepcopy(g_lr_scheduler.state_dict())
else:
    g_lr_scheduler_state = None
if d_lr_scheduler:
    d_lr_scheduler_state = copy.deepcopy(d_lr_scheduler.state_dict())
else:
    d_lr_scheduler_state = None    

state_dicts = {"g_state" : copy.deepcopy(generator.state_dict()),
               "d_state" : copy.deepcopy(discriminator.state_dict()),
               "g_optimizer_state" : copy.deepcopy(g_optimizer.state_dict()),
               "d_optimizer_state" : copy.deepcopy(d_optimizer.state_dict()),
               "g_lr_scheduler" : g_lr_scheduler_state,
               "d_lr_scheduler" : d_lr_scheduler_state} 

if not loading_checkpoint or without_checkpoint == "y":
    data_file = open(data_file_path, 'r')
    run_list = list(map(lambda x: int(x.split(",")[0]) if x[0] != "\n" else 0, 
                        data_file.readlines()))
    run = max(run_list) + 1
    data_file.close()

file_name = "{}_{}_{}_{}"
file_name = file_name.format(epoch_start, epoch_start+epochs, trainset, run)
print(file_name)
if os.path.isfile(os.path.join(state_dict_folder, file_name)):
    print("Name \"{}\" already exists\n".format(file_name))
    file_name = input("Give alternative name: ")

np.save(os.path.join(stats_folder, file_name), stats_to_save)
torch.save(state_dicts, os.path.join(state_dict_folder, file_name))
np.save(os.path.join(epoch_output_folder, file_name), epoch_outputs)

for name in hyperparams:
    hyperparams[name] = str(hyperparams[name])

data_file = open(data_file_path, 'a')
data_file.write("{}, {}, {}, {}, ".format(run, file_name, 
                                          epoch_start, epoch_start+epochs))
data_file.write(", ".join(hyperparams.values()))
data_file.write("\n")
data_file.close()      

for i in range(epochs):
  plt.figure()
  plt.imshow(g_epoch_outputs[i], 'gray')
  
plt.figure()
adv_loss_epoch_plot = np.concatenate((g_epoch_adv_loss.reshape((epochs, 1)),
                                      d_epoch_loss.reshape((epochs, 1)),
                                      d_epoch_real_loss.reshape((epochs, 1)),
                                      d_epoch_fake_loss.reshape((epochs, 1))
                                      ), axis=1)
plt.plot(adv_loss_epoch_plot, alpha=0.5)
plt.figure()
plt.plot(g_epoch_pixel_loss)
plt.figure()
adv_loss_batch_plot = np.concatenate((g_batch_adv_loss.reshape((batches*epochs, 1)),
                                      d_batch_loss.reshape((batches*epochs, 1)),
                                      d_batch_real_loss.reshape((batches*epochs, 1)),
                                      d_batch_fake_loss.reshape((batches*epochs, 1))
                                      ), axis=1)
plt.plot(adv_loss_batch_plot, alpha=0.5)
plt.figure()
plt.plot(g_batch_pixel_loss)
plt.figure()

with torch.set_grad_enabled(False):
    sample = data[0]
    plt.imshow(sample['lim'].reshape(512,512), 'gray')
    z = sample["lim"].reshape(1,1,512,512).to(device)
    generator.eval()
    gen_sino = generator(z.to(device)).detach().cpu().numpy().reshape(512,512)
    plt.figure()
    plt.imshow(gen_sino, 'gray')
    error = abs(gen_sino-sample['full'].numpy().reshape(512,512))
    plt.figure()
    plt.imshow(error, 'gray')
