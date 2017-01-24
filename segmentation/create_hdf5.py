# -*- coding: utf-8 -*-
#########################################################################
# Author: david
# Created Time: 2017年01月19日 星期四 10时14分10秒
# File Name: create_hdf5.py
# Description: 
#########################################################################
import random
from PIL import Image
#import Image
import numpy as np
import h5py
import glob
import cv2

# 图片目录
flist = glob.glob('./myroad/*.jpg')

datas = np.zeros((len(flist), 3, 300, 300))
labels = np.zeros((len(flist), 300, 300))

def createH5(mean):
    for i, _file in enumerate(flist):
        #print _file
        #if i != 0:
            #continue
        im = Image.open(_file)
        im = im.resize((300,300),Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
        # 转换成 BGR
        in_ = in_[:,:,::-1]
        # 减去均值
        in_ -= mean
        # 转换成(3, 300, 300)
        in_ = in_.transpose((2,0,1))
        datas[i, :, :, :] = in_

        name = _file.split('/')[2].split('.')
        _lfile  = './myroad/' + name[0] + '.mask.0.png'
        im = Image.open(_lfile)
        im = im.resize((300,300),Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
        #in_ = in_[:,:,0]
        #in_ -= np.array((104.00698793,116.66876762,122.67891434))
        #in_ = in_.transpose((2,0,1))
        labels[i, :, :] = in_
        
    # 写入 hdf5 文件
    with h5py.File('myroad_train.h5') as f:
        f['data'] = datas
        f['label'] = labels
        f.close()


def createMean():
    sum_image = None
    for i, _file in enumerate(flist):
        print _file
        image = cv2.imread(_file)
        img = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
        if sum_image is None:
            sum_image = np.ndarray(image.shape, dtype=np.float32)
            sum_image[:] = image
        else:
            sum_image += image
    mean = sum_image / 200
    res = np.array((0, 0, 0), dtype=np.float32)
    res[0] = np.mean(mean[:, :, 0])
    res[1] = np.mean(mean[:, :, 1])
    res[2] = np.mean(mean[:, :, 2])
    print res 
    return res 


if __name__ == '__main__':
    mean = createMean()
    createH5(mean)
    #from IPython import embed
    #embed()
    



