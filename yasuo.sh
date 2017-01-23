#!/bin/bash
#########################################################################
# Author: david
# Created Time: 2016年12月27日 星期二 11时00分50秒
# File Name: yasuo.sh
# Description: 
#########################################################################
for i in `find ./edgecolour -name "*.jpg"`;
do
    convert $i -resize 25% $i;
    echo "resize image $i to 25%"
done
