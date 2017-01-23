from xml.etree.ElementTree import parse                                                                                
import glob
import os,sys,cv2
import Image
import matplotlib.pyplot as plt
import pprint

_IMAGE_COPY_PATH = 'JPEGImages/'
_ANNO_PATH_='Annotations/'
li = sorted(glob.glob("./Annotations/*.xml"))
for filename in li:
    doc = parse(filename)
    root = doc.getroot()
    image_name,ext = os.path.splitext(filename.split('/')[2])
    image_name = image_name+'.jpg'
    img_path = os.path.join(_IMAGE_COPY_PATH, image_name)
    print img_path
    im = cv2.imread(img_path)
    # pprint.pprint(im)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
   
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
                          edgecolor='red', linewidth=3.5)
            )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig("result/" + image_name)
    plt.close('all')
plt.show()
