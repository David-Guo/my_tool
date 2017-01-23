#!/bin/bash
#########################################################################
# Author: david
# Created Time: Tue Dec 13 17:07:49 2016
# File Name: start.sh
# Description: 
#########################################################################
#nohup ./experiments/scripts/faster_rcnn_alt_opt.sh 0 ZF pascal_voc > `date +%m-%d-%H:%M:%S.log` 2>&1 &
rm ./data/cache/*
./experiments/scripts/faster_rcnn_alt_opt.sh 0 ZF pascal_voc
#nohup ./experiments/scripts/faster_rcnn_alt_opt.sh 0 ZF pascal_voc > out.log 2>&1 &
