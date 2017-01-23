#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
matplotlib.use("Agg") #edit by cayla
import _init_paths
from fast_rcnn.config import cfg, _merge_a_into_b
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import os.path as osp
import argparse
import glob
import pprint

#--------------config-------------
from easydict import EasyDict as edict

self_cfg = edict()
self_cfg.DRAW = False
self_cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
self_cfg.DATA_DIR = osp.abspath(osp.join(self_cfg.ROOT_DIR, 'data'))
self_cfg.MODELS_DIR = osp.abspath(osp.join(self_cfg.ROOT_DIR, 'models', 'pascal_voc'))


EXP_CLASS = 'corner'
self_cfg.CLASSES = ('__background__', EXP_CLASS)
DATASET = 'corner'
# xml path
# self_cfg._ANNO_PATH_ = self_cfg.DATA_DIR + '/' + DATASET + '/VOC2007/Annotations'
# image path
#self_cfg._JPEGImage = self_cfg.DATA_DIR + '/demo'
self_cfg._JPEGImage = self_cfg.DATA_DIR + '/' + DATASET + '/VOC2007/JPEGImages'
# result path
#self_cfg._RESULT = self_cfg.DATA_DIR + '/result'
self_cfg._RESULT = self_cfg.DATA_DIR + '/' + DATASET + '/VOC2007/gen_xml'
# net dict

NETS = {}
NETS['zf'] = ('ZF','ZF_faster_rcnn_final.caffemodel')

#_ANNO_PATH_= self_cfg._ANNO_PATH_
#---------------------------------
def gen_xml(class_name, dets, thresh=0.1, image_name=None):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print "fail !!!!!!!!!!!!!!!!!!!1"
        return
    
    # 生成根节点
    root = Element('annotation')
    
    for i in inds:
        bbox = dets[i, :4]
        # score = dets[i, -1]
        # 生成子节点
        obj = SubElement(root, 'object')
        obj_name = SubElement(obj, 'name')
        obj_name.text = class_name
        obj_pose = SubElement(obj, 'pose')
        obj_pose.text = 'Unspecified'
        obj_tru = SubElement(obj, 'truncated')
        obj_tru.text = '0'
        obj_diff = SubElement(obj, 'difficult')
        obj_diff.text = '0'
        obj_bb = SubElement(obj, 'bndbox')
        bb_xmin = SubElement(obj_bb, 'xmin')
        bb_xmin.text = str(int(bbox[0]))
        bb_ymin = SubElement(obj_bb, 'ymin')
        bb_ymin.text = str(int(bbox[1]))
        bb_xmax = SubElement(obj_bb, 'xmax')
        bb_xmax.text = str(int(bbox[2]))
        bb_ymax = SubElement(obj_bb, 'ymax')
        bb_ymax.text = str(int(bbox[3]))
    
    image_name = image_name.split('/')[-1].split('.')[0]
    filename = SubElement(root, 'filename')
    filename.text = image_name
    gen_xml_name = self_cfg._RESULT  + '/' + image_name + '.xml'
    tree = ElementTree(root)
    tree.write(gen_xml_name)


def vis_detections(im, class_name, dets, thresh=0.1 ,image_name = None):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print "fail !!!!!!!!!!!!!!!!!!!1"
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2)
            )
        #ax.text(bbox[0], bbox[1] - 2,
                #'{:s} {:.3f}'.format(class_name, score),
                #bbox=dict(facecolor='blue', alpha=0.5),
                #fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)

#=============================start==============================
    if self_cfg.DRAW:
        image_name = image_name.split('/')[-1].split('.')[0]
        xml = _ANNO_PATH_+'/'+image_name+'.xml'
        print(xml)

        if os.path.isfile(xml):
            doc = parse(xml)
            root = doc.getroot()
            objs = root.findall('object')
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                ax.add_patch(
                    plt.Rectangle((x1, y1),
                                  x2 - x1,
                                  y2 - y1, fill=False,
                                  edgecolor='cyan', linewidth=2)
                    )
#=============================end==============================
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(self_cfg.DATA_DIR, 'demo', image_name)

    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(self_cfg.CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        gen_xml(cls, dets, thresh=CONF_THRESH, image_name=image_name)
        # vis_detections(im, cls, dets, thresh=CONF_THRESH,image_name = image_name)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(self_cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(self_cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = sorted(glob.glob(self_cfg._JPEGImage + '/*'))
    #im_names = sorted(glob.glob(self_cfg._JPEGImage + '/*.jpg'))[:70]
    for im_name in im_names:
        res_name = im_name.split('/')[-1]
        print res_name
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
        # plt.savefig(self_cfg._RESULT +  "/res" + res_name)
        # plt.close('all')
    # plt.show()
    
