#!/bin/bash
#########################################################################
# Author: david
# Created Time: 2017年01月23日 星期一 14时01分54秒
# File Name: train.sh
# Description: 
#########################################################################
set -x
set -e

LOG="out.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python train.py
