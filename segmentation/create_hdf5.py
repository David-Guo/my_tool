# -*- coding: utf-8 -*-
#########################################################################
# Author: david
# Created Time: 2017年01月19日 星期四 10时14分10秒
# File Name: create_hdf5.py
# Description: 
#########################################################################
import random
from PIL import Image
import numpy as np
import h5py
import glob

flist = glob.glob('./image_2/*.png')

datas = np.zeros((len(flist), 3, 300, 300))
labels = np.zeros((len(flist), 300, 300))

for i, _file in enumerate(flist):
    #print _file
    im = Image.open(_file)
    im = im.resize((300,300),Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    datas[i, :, :, :] = in_

    name = _file.split('/')[2].split('_')
    _lfile  = './gt_image_2/' + name[0] + '_road_' + name[1]
    im = Image.open(_lfile)
    im = im.resize((300,300),Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32) / 255
    in_ = in_[:,:,2]
    #in_ -= np.array((104.00698793,116.66876762,122.67891434))
    #in_ = in_.transpose((2,0,1))
    labels[i, :, :] = in_
    

# 写入 hdf5 文件
with h5py.File('train.h5') as f:
    f['data'] = datas
    f['label'] = labels
    f.close()

if __name__ == '__main__':
    from IPython import embed
    embed()
    



