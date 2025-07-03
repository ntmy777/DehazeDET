import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

# dirs = ['train', 'test']
dirs = ['train']
classes = ['person',
'bicycle',
'car',
'motorbike',
'bus']

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(os.path.join(dir_path, 'Annotations', basename_no_ext + '.xml'))
    out_file = open(os.path.join(output_path, basename_no_ext + '.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        if xmlbox is None:
            continue
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

cwd = getcwd()

for dir_path in dirs:
    full_dir_path = cwd + '/' + dir_path 
    output_path = full_dir_path +'/VOC2007-FOG'
    print(dir_path, full_dir_path, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if dir_path == 'train':
        img_path = full_dir_path +  '/VOC2007-FOG'
        img_path_gt = full_dir_path +  '/JPEGImages'
    else:
        img_path = full_dir_path +  '/VOCtest-FOG'
        img_path_gt = full_dir_path +  '/JPEGImages'
    image_paths = getImagesInDir(img_path)
    image_paths_gt = getImagesInDir(img_path_gt)
    list_file = open(full_dir_path + '.txt', 'w')
    list_file_gt = open(full_dir_path + '_gt.txt', 'w')
    for image_path in image_paths:
        list_file.write(image_path + '\n')
        convert_annotation(full_dir_path, output_path, image_path)
        print('done')
    for image_path in image_paths_gt:
        list_file_gt.write(image_path + '\n')
        print('done gt')
        
    list_file.close()
    list_file_gt.close()
    print("Finished processing: " + dir_path)