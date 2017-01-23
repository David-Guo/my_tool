

1. 修改文件 ./experiments/cfgs/faster_rcnn_alt_opt.yml 文件中的三个路径为你的数据路径

2. 修改文件 _init_paths.py_ 中的 this_dir 指向你所在机器安装的 py-faster-cnn 路径中的 tools 目录

3. 修改文件 ./tools/train_faster_rcnn_alt_opt.py ./tools/test_net.py 前几行中的 classes 和 DEVKIT_PATH 为你的实验数据目录

**也可以再 train_faster_rcnn_alt_opt 中修改迭代次数**

4. 在 ./data 目录下 ln 链接 imagenet 模型文件到 ./data/imagenet_models 

5. 在 ./data 目录下 ln 链接 数据集到 ./data 目录

数据集目录树与原 py-faster-cnn 相似，只是修改了 voc2007devkit 为你的自己的实验名，例如下面是 data 目录下的 fence 实验


    fence
    ├── annotations_cache
    ├── results
    │   └── VOC2007
    │       └── Main
    └── VOC2007
        ├── Annotations
        ├── ImageSets
        │   └── Main
        └── JPEGImages
