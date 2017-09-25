#! /usr/bin/env python
# coding: utf-8

"""
Calc dot product for a certain page.

Please configure in [DOT] section of config file.
    PAGE = 32
    THRESHOLD = -3
    VIDEO_FILEPATH = ./out/dot_3.mp4

    PAGE :           page to be calculate cumulative flow.
    THRESHOLD :      If a pixel has dot product value under THRESHOLD, it is drawn with a red circle in video.
    DUMP_FILEPATH :  filepath to dump dotProduct_arr

    VIDEO_FILEPATH : filepath to output video. only used when  DEFAULT.OUTPUT_VIDEO = yes
"""

import numpy as np
import sys
import time
from matplotlib import pyplot as plt
import cv2
from configparser import ConfigParser

import ciputil

TIME_MAX = None
PAGE = None
OUTPUT_VIDEO = None

def calc_dot_product(fixDirection_arr):
    """
    return dotProduct_arr[time][960][960], 1 origin time
    """

    def dot_product(flow):
        """
        return matrix[960][960] of minimum dot product for each pixel
        """
        height = flow.shape[0]
        width = flow.shape[1]
        dotProductNeighbor = np.zeros ((10, height + 2, width + 2))

        flowMargin = np.zeros((height + 2, width + 2, 2))
        flowMargin[1:-1, 1:-1] = flow

        shiftIterator = 0
        for xShift in range(0, 2):
            for yShift in range(-1, 2):
                if (xShift != 0 or yShift != 0):
                    shifted = np.zeros((height + 2, width + 2, 2))
                    shifted[1 + xShift : height + xShift + 1, 1 + yShift : width + yShift + 1] = flow
                    dotProductNeighbor[shiftIterator] = np.sum(flowMargin * shifted, axis=2)
                    dotProductNeighbor[shiftIterator + 1][1:-1, 1:-1] \
                        = dotProductNeighbor[shiftIterator][1 + xShift : height + xShift + 1, 1 + yShift : width + yShift + 1]
                    shiftIterator += 2
        dotProduct = np.min(dotProductNeighbor, axis=0)

        return dotProduct[1:-1,1:-1]

    prevImg = ciputil.get_image(time=1, page=PAGE)
    prevFixImg = ciputil.get_stabilized_image(prevImg, fixDirection_arr[PAGE][1])

    dotProduct_arr = np.zeros((TIME_MAX + 1, 960, 960))

    for time in range(2, TIME_MAX + 1):
        nextImg = ciputil.get_image(time=time, page=PAGE)
        nextFixImg = ciputil.get_stabilized_image(nextImg, fixDirection_arr[PAGE][time])

        denseFlow = ciputil.calc_dense_flow(prevFixImg, nextFixImg)
        dotProduct_arr[time] = dot_product(denseFlow)

    return dotProduct_arr

def output_dot_video(dotProduct_arr, dotProductThreshold, fixDirection_arr, videoFilepath):
    """
    output a video with circle at small (big minus) dot product pixel
    """

    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    video = cv2.VideoWriter(videoFilepath, fourcc, 5.0, (960, 960))

    for time in range(1, TIME_MAX + 1):
        print("time:{}".format(time))
        img = ciputil.get_image(time=time, page=PAGE)
        stabImg = ciputil.get_stabilized_image(img, fixDirection_arr[PAGE][time])

        reversePoint_arr = np.where(dotProduct_arr[time] < dotProductThreshold)

        dotImg = stabImg
        for i in range(len(reversePoint_arr[0])):
            x = reversePoint_arr[0][i]
            y = reversePoint_arr[1][i]
            dotImg = cv2.circle(dotImg, (x, y), 8, (255,0,0), -1)
        video.write(dotImg)
    video.release()

def main():
    global TIME_MAX
    global PAGE

    configFilepath = "./config/config.ini"
    TIME_MAX, _, OUTPUT_VIDEO = ciputil.read_config(configFilepath)
    PAGE, threshold, dumpFilepath, videoFilepath = ciputil.read_config_dot(configFilepath)

    fixDirection_arr=np.load("./out/fixDir.npy")

    print("START: calculating dense flow and dot product")
    dotProduct_arr = calc_dot_product(fixDirection_arr)
    np.save(dumpFilepath, dotProduct_arr)
    print("DONE: dump to {}".format(dumpFilepath))

    if OUTPUT_VIDEO:
        output_dot_video(dotProduct_arr, threshold, fixDirection_arr,videoFilepath)
        print("DONE: output video to {}".format(videoFilepath))

if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print("\nelapse time: {} sec".format(elapse))
