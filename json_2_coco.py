# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 9:41  2020-12-16

import json
import os
import numpy as np
import skimage.draw as draw
import scipy.misc as misc
import cv2
import matplotlib.pyplot as plt
from labelme import utils
import PIL.Image
import argparse
import logging
from glob import glob
from pycocotools.coco import COCO


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

parser = argparse.ArgumentParser(description='labeled json convert to coco format')
parser.add_argument('--data_dir', default=r"D:\MyNAS\SynologyDrive\Demo_Test\label2coco_convert\balloon\val", type=str,
                    help='image data path')
parser.add_argument('--data_json',
                    default=r"D:\MyNAS\SynologyDrive\Demo_Test\label2coco_convert\balloon\val\via_region_data.json", type=str,
                    help='via or labeme json file path')
parser.add_argument('--save_json', default=r"D:\MyNAS\SynologyDrive\Demo_Test\label2coco_convert\balloon\cocoval.json",
                    type=str,
                    help='conver coco json format save path')
parser.add_argument('--show', default=True, type=bool,
                    help='if show the convert result')
parser.add_argument('--label_tool', default="via", choices=["via", "labelme"], type=str,
                    help='choose label tool')

config = parser.parse_args()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def Load_Json(path):
    data = json.load(open(path))
    return data


class via2coco(object):
    def __init__(self, jsondatas, datadir, savejson):
        self.jsondatas = jsondatas
        self.datadir = datadir
        self.savejson = savejson
        self.id = 0
        self.cat_id = 1
        self.dataName = "tree"
        self.coco = dict()
        self.coco['info'] = {"description": None,
                             "url": None,
                             "version": None,
                             "year": None,
                             "contributor": None,
                             "date_created": None
                             }
        self.coco["images"] = []
        self.coco['annotations'] = []
        self.coco["categories"] = []
        for num, json_file in enumerate(self.jsondatas):
            logging.info("convert json: {}".format(json_file))
            data = Load_Json(json_file)
            data = list(data.values())
            for num, json_file in (enumerate(data)):
                img_id = json_file['filename']
                full_imgpath = os.path.join(self.datadir, img_id)
                if not os.path.isfile(full_imgpath):
                    continue
                regions = json_file['regions']
                for i in range(len(regions)):
                    self.id += 1
                    if isinstance(regions, list):
                        regions_i = regions[i]
                    if isinstance(regions, dict):
                        regions_i = regions[str(i)]
                    bbox, h, w = self.Get_bbox(regions_i, img_id)
                    self.addAnnoItem(regions_i, img_id, bbox, self.id, self.cat_id)
                    self.assCategories(self.cat_id)
                    self.addImages(img_id, h, w)

        ### save coco format json
        json.dump(self.coco, open(self.savejson, 'w'), cls=MyEncoder)

    def assCategories(self, label):
        categorie = {}
        categorie['supercategory'] = self.dataName
        categorie['id'] = label  # 0 默认为背景
        categorie['name'] = self.dataName
        self.coco["categories"].append(categorie)

    def addImages(self, img_id, height, width):
        image_item = dict()
        image_item["file_name"] = img_id
        image_item["height"] = height
        image_item["width"] = width

        imgid = img_id.split(".")[0]
        image_item["id"] = imgid
        self.coco["images"].append(image_item)

    def addAnnoItem(self, regions, image_id, bbox, annotation_id, category_id):

        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        for x, y in zip(regions['shape_attributes']['all_points_x'], regions['shape_attributes']['all_points_y']):
            seg.append([x, y])
        annotation_item['segmentation'] = [list(np.asarray(seg).flatten())]
        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0

        imgid = image_id.split(".")[0]
        annotation_item['image_id'] = imgid
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        annotation_item['id'] = annotation_id
        self.coco['annotations'].append(annotation_item)

    def Get_bbox(self, p, imgid):
        imgpath = os.path.join(self.datadir, imgid)
        # data = misc.imread(imgpath)
        data = cv2.imread(imgpath)
        h, w = data.shape[:2]
        del data

        mask = np.zeros((h, w))
        rr, cc = draw.polygon(p['shape_attributes']['all_points_y'], p['shape_attributes']['all_points_x'])
        mask[rr, cc] = 1

        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        # #
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        width, height = y2 - y1, x2 - x1

        # cv2.rectangle(data,(x1,y1), (x1+height,y1+width),(255,0,0))
        # plt.subplot(121)
        # plt.imshow(data)
        # plt.show()
        return [x1, y1, height, width], h, w


class Convert_Show:
    def __init__(self, savejson, datadir):
        self.savejson = savejson
        self.datadir = datadir
        self.coco = COCO(savejson)
        # cats = self.coco.loadCats(self.coco.getCatIds())
        self.Show()

    def Show(self):
        imgIds = self.coco.getImgIds()
        for idx, k in enumerate(imgIds):
            k = imgIds[idx]
            img = self.coco.loadImgs([k])[0]

            I = cv2.imread('%s/%s' % (self.datadir, img['file_name']))
            I = I[:, :, [2, 1, 0]]
            # 加载和可视化instance标注信息
            catIds = []
            for ann in self.coco.dataset['annotations']:
                if ann['image_id'] == k:
                    catIds.append(ann['category_id'])

            plt.imshow(I)
            plt.axis('off')
            annIds = self.coco.getAnnIds(imgIds=[img['id']], catIds=catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            self.coco.showAnns(anns)
            plt.show()


class labelme2coco:
    def __init__(self, labelme_json=[], save_json_path=None):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.save_json()

    def data_transfer(self):

        for num, json_file in enumerate(self.labelme_json):
            logging.info("convert json: {}".format(json_file))
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num))
                for shapes in data['shapes']:
                    label = shapes['label']
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    points = shapes['points']  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points.append([points[0][0], points[1][1]])
                    points.append([points[1][0], points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data['imageData'])
        height, width = img.shape[:2]
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]
        self.height = height
        self.width = width
        return image

    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'none'
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()

        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)


def startFormatConvert(arg):
    data_dir = arg.data_dir
    data_json = arg.data_json
    save_json = arg.save_json
    label_tool = arg.label_tool
    if label_tool == "via":
        logging.info("start via to coco format convert")
        if os.path.isdir(data_json):
            via_jsons = glob(data_json + '/*.json')
        elif os.path.isfile(data_json):
            via_jsons = [data_json]
        else:
            logging.error("check json file")
            exit(0)

        if len(via_jsons) == 0:
            logging.warning("find 0 json files in path : {}".format(data_json))
            exit(0)
        via2coco(via_jsons, data_dir, save_json)
        logging.info("convert finished !")

    if label_tool == "labelme":
        logging.info("start labelme to coco format convert")
        labelme_jsons = glob(data_json + '/*.json')
        if len(labelme_jsons) == 0:
            logging.warning("find 0 json files in path : {}".format(data_json))
            exit(0)
        labelme2coco(labelme_jsons, save_json)
        logging.info("convert finished !")

    if arg.show:
        logging.info("show convert result .....")
        Convert_Show(save_json, data_dir)
        logging.info("show finished !")


if __name__ == '__main__':
    if config.data_dir is None or config.data_json is None:
        logging.info("data path error and will be exit .")
        exit(0)

    # with open(config.data_json, "r") as fid:
    #     data = json.load(fid)
    #
    # newdata = dict()
    # for idx,item in enumerate(data.keys()):
    #     print(item)
    #     value = data[item]
    #     fileName = value['filename']
    #     newName = str(idx).zfill(5)+ ".jpg"
    #     newKey = item.replace(fileName, newName)
    #
    #     value['filename'] = newName
    #     oldImgName = os.path.join(config.data_dir, fileName)
    #     newImgName = os.path.join(config.data_dir, newName)
    #
    #     os.rename(oldImgName, newImgName)
    #     newdata[newKey] = value
    #
    # json.dump(newdata, open(config.data_json.replace(".json", "rename_.json."), 'w'), indent=4, cls=MyEncoder)
    config.data_json = r"D:\MyNAS\SynologyDrive\Demo_Test\label2coco_convert\balloon\val\via_region_datarename_.json"
    startFormatConvert(config)

