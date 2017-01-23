#!/bin/bash
#########################################################################
# Author: david
# Created Time: Sat Dec 24 09:53:03 2016
# File Name: create_pack.sh
# Description: 
#########################################################################
RCNN_PATH=/data/ai/code//py-faster-rcnn

mkdir data
mkdir data/cache
ln -s $RCNN_PATH/data/imagenet_models data/imagenet_models
ln -s $RCNN_PATH/data/VOCdevkit2007 data/VOCdevkit2007

cp -r $RCNN_PATH/experiments experiments
cp -r $RCNN_PATH/models models

cp -r $RCNN_PATH/tools/ tools
