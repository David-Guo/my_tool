#!/bin/bash
#########################################################################
# Author: david
# Created Time: Thu Dec 22 09:12:37 2016
# File Name: test.sh
# Description: 
#########################################################################

time ./tools/test_net.py --gpu 0 \
    --def models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt \
    --net output/faster_rcnn_alt_opt/voc_2007_trainval/ZF_faster_rcnn_final.caffemodel \
    --imdb voc_2007_test \
    --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
