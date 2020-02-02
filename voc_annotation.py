import xml.etree.ElementTree as ET
import os

# classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
# 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCAnnotation(object):
    def __init__(self, data_path, year, mode, class_names_path):
        """

        :param year: voc 年份
        :param mode: train or val
        :param data_path: 数据存放目录  ~/segment_data 只需到VOCdevkit目录上一层
        :param classes_path: 类别数据存放地址 data/voc_classes.txt
        """
        self.class_names = self._class_names(class_names_path)
        self._parse_voc_data(year, mode, data_path)

    def _parse_voc_data(self, year, mode, data_path):
        save_path = './data/%s_%s.txt' % (year, mode)
        if os.path.exists(save_path):
            return
        image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (data_path, year, mode)).read().strip().split()
        list_file = open(save_path, 'w')
        for image_id in image_ids:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (data_path, year, image_id))
            self._convert_annotation(data_path, year, image_id, list_file)
            list_file.write('\n')
        list_file.close()

    def _class_names(self, class_names_path):
        class_names = open(class_names_path).readlines()
        class_names = [class_name.strip() for class_name in class_names]
        return class_names

    def _convert_annotation(self, data_path, year, image_id, list_file):
        in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml' % (data_path, year, image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.class_names or int(difficult) == 1:
                continue
            cls_id = self.class_names.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            # 将索引从1开始 0表示背景
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id + 1))
