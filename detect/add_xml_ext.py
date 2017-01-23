# -*- coding: utf-8 -*-
#########################################################################
# Author: david
# Created Time: 2016年12月23日 星期五 19时36分43秒
# File Name: add_xml_ext.py
# Description: 给所有文件加上扩展名
#########################################################################
import glob
import shutil

flist = glob.glob('./*')


for i in flist:
    src = i.split('/')[1]
    dest = src + '.xml'
    shutil.copyfile(src, dest)

