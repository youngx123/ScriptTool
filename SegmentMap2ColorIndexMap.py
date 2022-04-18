# -*- coding: UTF-8 -*-
# Author: Young
# Time: 17:32 2021/2/1

import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from random import randint
from PIL import Image

color_dict = {0: [0, 0, 0],
              1: [0, 0, 170], 2: [0, 0, 255],
              3: [0, 85, 255], 4: [0, 170, 255],
              5: [0, 255, 255], 6: [85, 255, 170],
              7: [170, 255, 85], 8: [255, 255, 0],
              9: [255, 170, 0], 10: [255, 85, 0],
              11: [255, 0, 0], 12: [170, 0, 0],
              13: [], 14: [],
              }


def randomPalette(length, min, max):
    return [randint(min, max) for x in range(length)]


def indexcolor(c, categoryValue):
    for i in range(0, categoryValue):
        rgb = color_dict[i]
        c += [rgb]
    c[:3 * categoryValue] = np.array(c, dtype='uint8').flatten()
    return c


if __name__ == '__main__':
    path1 = "label_png"
    save = "./index_seg/"

    path_names = glob(path1 + "/*.png")
    categoryValue = 13
    for path in path_names:
        img = Image.open(path)

        index_color = img.copy()
        index_color = index_color.convert("P")
        c = []
        c = indexcolor(c, categoryValue)
        # ## color image
        index_color.putpalette(c)

        # # seg index value
        # index_arr = np.array(index_color)
        index_color.save(save + os.path.basename(path))
        # misc.imsave(save + os.path.basename(path), index_color) ## segment map
