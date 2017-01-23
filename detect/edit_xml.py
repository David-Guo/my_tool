#!/usr/bin/python
#########################################################################
# Author: david
# Created Time: Wed Dec 14 19:56:47 2016
# File Name: edit_xml.py
# Description: 批量修改 xml 工具
#########################################################################

from xml.etree.ElementTree import parse                                                                                
import glob


li = glob.glob("./Annotations/*.xml")

for filename in li:
    doc = parse(filename)
    root = doc.getroot()
    root.find('size').find('width').text = str(int(root.find('size').find('width').text)/4)
    root.find('size').find('height').text = str(int(root.find('size').find('height').text)/4)

    objs = root.findall('object')
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        bbox.find('xmin').text = str(int(bbox.find('xmin').text)/4)
        bbox.find('ymin').text = str(int(bbox.find('ymin').text)/4)
        bbox.find('xmax').text = str(int(bbox.find('xmax').text)/4)
        bbox.find('ymax').text = str(int(bbox.find('ymax').text)/4)
   
    doc.write(filename)
