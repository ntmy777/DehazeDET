import glob
import os
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
from PIL import Image

dirs = ['test_foggy']
class_mapping = {
    0: 2,   # car in foggy_driving maps to car in VOC
    1: 0,   # person in foggy_driving maps to person in VOC
    2: 1,   # bicycle in foggy_driving maps to bicycle in VOC
    3: 4,   # bus in foggy_driving maps to bus in VOC
    6: 3    # motorcycle in foggy_driving maps to motorcycle in VOC
}
def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append(filename)
    return image_list

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

    
def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]
    in_file = open(os.path.join(dir_path, 'Annotations', basename_no_ext + '.txt'))
    out_file = open(os.path.join(output_path, basename_no_ext + '.txt'), 'w')
    
    # Read the annotation lines
    lines = in_file.readlines()
    
    size = get_image_size(image_path)
    w, h = size
    
    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        if cls_id not in class_mapping:
            continue
        cls_id_voc = class_mapping[cls_id]
        x_min, y_min, x_max, y_max = map(float, parts[1:])
        bb = convert(size, (x_min, x_max, y_min, y_max))
        out_file.write(str(cls_id_voc) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()
cwd = getcwd()
for dir_path in dirs:
    full_dir_path = cwd + '\\' + dir_path 
    output_path = full_dir_path + '\\VOC2007-FOG'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    img_path = full_dir_path + '\\JPEGImages'
    print("Converting ", dir_path, " ...")
    print(img_path)
    image_paths = getImagesInDir(img_path)
    list_file = open(full_dir_path + '.txt', 'w')
    
    for image_path in image_paths:
        list_file.write(image_path + '\n')
        convert_annotation(full_dir_path, output_path, image_path)
        
    list_file.close()
    print("Finished processing: " + dir_path)