#! /usr/bin/env python
# coding: utf-8

"""
stabilize image using sparse optical flow.
dump numpy files of 3d(x, y, z) image movement direction.

configure in [STABILIZE] section of config/config.ini file.
    ANGLE_THRESH: If the variance of the flows in a frame is under ANGLE_THRESH, features in the frame are thought to be undetected.
    DUMP_FILEPATH: path to which dump numpy files (fixDirection_arr)
    VIDEO_FILEPATH: path to which output stabilized video
"""

import sys
import time
import numpy as np
import statistics as st
import cv2
import matplotlib.pyplot as plt
from configparser import ConfigParser
import ciputil

TIME_MAX = None
PAGE_MAX = None
OUTPUT_VIDEO = None

class FeatureError(Exception):
    def __init__(self, value):
        self.value = value

def calc_fix_direction(angleThresh):
    """
    calculate fix direction for at each time point.
    fixDirection_arr[page][time] = fixDirectionX, fixDirectionY, fixDirectionZ
    (page: 1 ~ PAGE_MAX, time: 1 ~ TIME_MAX)
    """
    def get_feature(prevImg, nextImg, prevFeature):
        assert prevImg.shape == nextImg.shape
        #parameter of Lucas-Kanade method
        lk_params = dict(winSize = (20, 20),
                            maxLevel = 5,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if len(prevImg.shape) == 3:
            prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
            nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
            nextFeature, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevFeature, None, **lk_params)
            prevFeatureFiltered = prevFeature[status == 1]
            nextFeatureFiltered = nextFeature[status == 1]
        return prevFeatureFiltered, nextFeatureFiltered

    def get_binarization(gray):
        """
        The threshold is determined by the Otus algorithm
        """
        _, binaryImg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binaryImg

    def calc_sparseFlow(prevFeatureFiltered, nextFeatureFiltered):
        sparseFlow = np.zeros((nextFeatureFiltered.shape[0],3))
        for i, (prevPoint, nextPoint) in enumerate(zip(prevFeatureFiltered, nextFeatureFiltered)):
            prevX, prevY = prevPoint.ravel()
            nextX, nextY = nextPoint.ravel()
            sparseFlow[i][0] = nextX - prevX
            sparseFlow[i][1] = nextY - prevY
        return sparseFlow

    def calc_movement(sparseFlow):
        try:
            meanX, meanY, meanZ = np.mean(sparseFlow, axis=0)
        except ZeroDivisionError:
            meanX, meanY, meanZ = 0, 0, 0
        movement = np.array([meanX, meanY, meanZ])
        return movement

    def calc_angle_variance(sparseFlow):
        filterSparseFlow = sparseFlow.copy()
        filterSparseFlow[filterSparseFlow <= 5] = 0
        filterSparseFlowX = filterSparseFlow[:, 0]
        filterSparseFlowY = filterSparseFlow[:, 1]
        angle_arr = np.arctan2(filterSparseFlowX, filterSparseFlowY) * 180 / np.pi
        angleVar = np.var(angle_arr)
        return angleVar

    def normalized_variance(angleVar_arr):
        varMax = np.amax(angleVar_arr)
        if varMax != 0:
            normAngleVar_arr = angleVar_arr / varMax
        else:
            return angleVar_arr
        return normAngleVar_arr


    fixDirection_arr = np.zeros((PAGE_MAX + 1,TIME_MAX + 1, 3))  # +1 to adjust to 1 origin of time
    allPage_fixDirection_arr = np.zeros((TIME_MAX + 1, 3))
    latestMovement = np.array([0, 0, 0])
    angleVar_arr = np.zeros((PAGE_MAX+1, TIME_MAX+1))
    movement_arr = np.zeros((PAGE_MAX + 1, TIME_MAX + 1, 3))

    feature_params = dict(maxCorners = 200,
                            qualityLevel = 0.001,
                            minDistance = 10,
                            blockSize = 5)

    for time in range(2, TIME_MAX+1):
        for page in range(1, PAGE_MAX + 1):
            prevImg = ciputil.get_image(time=time-1, page=page)
            prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
            nextImg = ciputil.get_image(time=time, page=page)
            flowMask = get_binarization(prevGray)
            prevFeature = cv2.goodFeaturesToTrack(prevGray, mask=flowMask, **feature_params)
            try:
                prevFeatureFiltered, nextFeatureFiltered = get_feature(prevImg, nextImg, prevFeature)
                if prevFeatureFiltered.shape[0] <= 50:
                    raise FeatureError("Not detect feature")
                sparseFlow = calc_sparseFlow(prevFeatureFiltered, nextFeatureFiltered)
                prevFeature = nextFeatureFiltered.reshape(-1, 1, 2)
                angleVar = calc_angle_variance(sparseFlow)
                angleVar_arr[page][time] = angleVar
                movement_arr[page][time] = calc_movement(sparseFlow)
            except FeatureError:
                movement_arr[page][time] = movement_arr[page-1][time]

    normAngleVar_arr = normalized_variance(angleVar_arr)
    for time in range(2, TIME_MAX+1):
        for page in range(1, PAGE_MAX+1):
            if normAngleVar_arr[page][time] >= angleThresh:
                movement_arr[page][time] = movement_arr[page-1][time]
            for i in range(3):
                fixDirection_arr[page][time][i] = fixDirection_arr[page][time - 1][i] + movement_arr[page][time][i]
    return fixDirection_arr


def output_stabilized_video(fixDirection_arr, videoFilepath, configFilepath):
    """
    output stabilized video according to fixDirection_arr
    """

    config = ConfigParser()
    config.read(configFilepath)
    if config["VIDEO"]["PAGE"] == "ALL":
        pageFirst = 1
        pageLast = PAGE_MAX
    else:
        pageFirst = int(config["VIDEO"]["PAGE"])
        pageLast = pageFirst
    print("START: output video to {}, pageFirst: {}, pageLast: {}".format(videoFilepath, pageFirst, pageLast))

    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    video = cv2.VideoWriter(videoFilepath, fourcc, 5.0, (960, 960))
    waitImg = np.zeros((960, 960, 3), np.uint8)  # image for waiting

    for page in range(pageFirst, pageLast + 1):
        for time in range(1, TIME_MAX + 1):
            img = ciputil.get_image(time=time, page=page)
            fixImg = ciputil.get_stabilized_image(img, fixDirection_arr[page][time])

            data = "[page: {0:03d} time: {1:03d}]".format(page, time)
            cv2.putText(fixImg, data, (160, 935),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

            video.write(fixImg)
        for _ in range(10):
            video.write(waitImg)
    video.release()

    print("DONE: output video to {}".format(videoFilepath))


def main():
    global TIME_MAX
    global PAGE_MAX
    global OUTPUT_VIDEO

    configFilepath = "./config/config.ini"
    TIME_MAX, PAGE_MAX, OUTPUT_VIDEO = ciputil.read_config(configFilepath)
    angleThresh, dumpFilepath, videoFilepath = ciputil.read_config_stabilize(configFilepath)
    fixDirection_arr = calc_fix_direction(angleThresh)
    np.save(dumpFilepath, fixDirection_arr)
    print("DONE:  calcurate fix direction")

    if OUTPUT_VIDEO:
        output_stabilized_video(fixDirection_arr, videoFilepath, configFilepath)


if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print('\nelapse time: {} sec'.format(elapse))
