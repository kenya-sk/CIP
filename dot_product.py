#! /usr/biin/env python
# coding: utf-8

"""
caluculate dot product for a certain page.
dump numpy file of dot product of cumulative flows.

configure [DOT] section in config file.
    PAGE           : page for which calculate cumulative flow.
    FLOW_THRESHOLD : If a pixel has flow with norm under FLOW_THRESHOLD, the flow is rounded to (0,0).
    DOT_THRESHOLD  : If a pixel has dot product value under DOT_THRESHOLD, it is drawn with a red circle in video. only used when DEFAULT.OUTPUT_VIDEO = yes.
    WINDOW_SIZE    : The size of neibor pixel window. Set ODD NUMBER and GREATOR THAN 3.
    RECALCULATE    : Whether recalculate the dot product array or not.
    DUMP_FILEPATH  : filepath to dump dotProduct_arr
    VIDEO_FILEPATH : filepath to output video. only used when DEFAULT.OUTPUT_VIDEO = yes
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

def calc_dot_product(cmlFlow_arr, windowSize, flowThreshold):
    """
    return dotProduct_arr[time][960][960], 1 origin time
    """

    assert windowSize%2 == 1
    assert windowSize >= 3

    def dot_product(flow, windowSize, flowThreshold):
        """
        return matrix[960][960] of minimum dot product for each pixel
        """

        assert flow.shape[0] == flow.shape[1]
        width = flow.shape[0]
        margin = int(windowSize/2)
        widthMargin = margin + width + margin #two direction +, -

        flow[np.sum(flow*flow, axis=2) < flowThreshold]=[0,0]
        flowMargin = np.zeros((widthMargin, widthMargin, 2))
        flowMargin[margin:-margin, margin:-margin] = flow

        neighborNum = ((margin + 1)*(margin*2 + 1) - 1)*2 #((num of xShift) * (num of yShift) - (0,0)) * (reverse direction)
        dotProductNeighbor = np.zeros ((neighborNum, widthMargin, widthMargin))
        shiftIterator = 0
        for xShift in range(margin + 1):
            for yShift in range(-margin, margin + 1):
                if (xShift != 0 or yShift != 0):
                    shifted = np.zeros((widthMargin, widthMargin, 2))
                    shifted[margin + xShift : margin + width + xShift, margin + yShift : margin + width + yShift] = flow
                    dotProductNeighbor[shiftIterator] = np.sum(flowMargin * shifted, axis=2)
                    dotProductNeighbor[shiftIterator + 1][margin:-margin, margin:-margin] \
                        = dotProductNeighbor[shiftIterator][margin + xShift : margin + width + xShift, margin + yShift : margin + width + yShift]
                    shiftIterator += 2
        assert shiftIterator == neighborNum

        dotProduct = np.min(dotProductNeighbor, axis=0)
        return dotProduct[margin:-margin,margin:-margin]

    dotProduct_arr = np.zeros((TIME_MAX + 1, 960, 960))
    for time in range(2, TIME_MAX + 1):
        print(time)
        cmlFlow = cmlFlow_arr[time]
        dotProduct_arr[time] = dot_product(cmlFlow, windowSize, flowThreshold)
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

        divisionPoint_arr = np.array(np.array(np.where(dotProduct_arr[time] < dotProductThreshold)))

        dotImg = stabImg
        assert divisionPoint_arr.shape[0] == 2
        numPoints = divisionPoint_arr.shape[1]
        for i in range(numPoints):
            x = divisionPoint_arr[1][i]
            y = divisionPoint_arr[0][i]
            dotImg = cv2.circle(dotImg, (x, y), 3, (255,0,0), -1)
        video.write(dotImg)
    video.release()

def main():
    global TIME_MAX
    global PAGE

    configFilepath = "./config/config.ini"
    TIME_MAX, PAGE_MAX, OUTPUT_VIDEO = ciputil.read_config(configFilepath)

    PAGE, windowSize, cmlFlowFilepath, videoFilepath = ciputil.read_config_cumulative(configFilepath)
    cmlFlow_arr = np.load(cmlFlowFilepath)
    recalculate, windowSize, flowThreshold, dotThreshold, dumpFilepath, videoFilepath = ciputil.read_config_dot(configFilepath)

    for page in range(1, PAGE_MAX+1):
        PAGE = page

        if recalculate:
            print("START: calculating dot product")
            cmlFlowFilepath = "./out/cml_{}.npy".format(page)
            cmlFlow_arr = np.load(cmlFlowFilepath)
            dotProduct_arr = calc_dot_product(cmlFlow_arr, windowSize, flowThreshold)
            dumpFilepath = "./out/dot_{}.npy".format(page)
            np.save(dumpFilepath, dotProduct_arr)
            print("DONE: dump to {}".format(dumpFilepath))
        else:
            dotProduct_arr = np.load(dumpFilepath)
            print("DONE: load dot product array from {}".format(dumpFilepath))

        if OUTPUT_VIDEO:
            _, fixDirectionFilepath, _ = ciputil.read_config_stabilize(configFilepath)
            fixDirection_arr = np.load(fixDirectionFilepath)
            videoFilepath = "./out/dot_{}.mp4".format(page)
            output_dot_video(dotProduct_arr, dotThreshold, fixDirection_arr,videoFilepath)
            print("DONE: output video to {}".format(videoFilepath))

if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print("\nelapse time: {} sec".format(elapse))
