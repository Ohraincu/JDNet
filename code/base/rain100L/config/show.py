import os
import sys
import cv2
import argparse
import math
import numpy as np
import itertools

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable 
from torch.utils.data import DataLoader

import settings
from dataset import ShowDataset
from model import ODE_DerainNet 
from cal_ssim import SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
#torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) 

class Session:
    def __init__(self):
        self.show_dir = settings.show_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.show_dir)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % settings.show_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        if len(settings.device_id) >1:
            self.net = nn.DataParallel(ODE_DerainNet()).cuda()
            #self.l2 = nn.DataParallel(MSELoss(),settings.device_id)
            #self.l1 = nn.DataParallel(nn.L1Loss(),settings.device_id)
            #self.ssim = nn.DataParallel(SSIM(),settings.device_id)
            #self.vgg = nn.DataParallel(VGG(),settings.device_id)
        else:
            torch.cuda.set_device(settings.device_id[0])
            self.net = ODE_DerainNet().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}
        self.ssim=SSIM().cuda()
        self.a=0
        self.t=0
    def get_dataloader(self, dataset_name):
        dataset = ShowDataset(dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
    def inf_batch(self, name, batch,i):
        O, B, file_name= batch['O'].cuda(), batch['B'].cuda(), batch['file_name']
        file_name = str(file_name[0])
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        with torch.no_grad():
            import time
            t0=time.time()
            derain = self.net(O)
            t1 = time.time()
            comput_time=t1-t0
            print(comput_time)
            ssim = self.ssim(derain, B).data.cpu().numpy()
            psnr = PSNR(derain.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
            print('psnr:%4f-------------ssim:%4f'%(psnr, ssim))
            return derain, psnr, ssim, file_name

    def save_image(self, No, imgs, name, psnr, ssim, file_name):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape

            img_file = os.path.join(self.show_dir, '%s.png' % (file_name))
            print(img_file)
            cv2.imwrite(img_file, img)


def run_show(ckp_name):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    sess.net.eval()
    dataset = 'test'
    if settings.pic_is_pair is False:
        dataset = 'train-w'
    dt = sess.get_dataloader(dataset)

    for i, batch in enumerate(dt):
        logger.info(i)
        if i>-1:
            imgs,psnr,ssim, file_name= sess.inf_batch('test', batch,i)
            sess.save_image(i, imgs, dataset, psnr, ssim, file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest_net')

    args = parser.parse_args(sys.argv[1:])
    
    run_show(args.model)

