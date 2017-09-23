#! /usr/bin/env python
# coding: utf-8

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

class Feature(Exception):
    def __init__(self, value):
        self.value = value

def calc_fix_direction():
    """
    calculate fix direction for at each time point.
    fixDirection_arr[page][time][xyz] = [1-PAGE_MAX+1][1-TIME_MAX+1](fixDirectionX, fixDirectionY, fixDirectionZ)
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
            prevGood = prevFeature[status == 1]
            nextGood = nextFeature[status == 1]
        return prevGood, nextGood

    def calc_flow(prevGood, nextGood):
        flow = np.zeros((3,nextGood.shape[0]))
        for i, (nextPoint, prevPoint) in enumerate(zip(nextGood, prevGood)):
            prevX, prevY = prevPoint.ravel()
            nextX, nextY = nextPoint.ravel()
            flow[0][i] = nextX - prevX
            flow[1][i] = nextY - prevY
        return flow

    def calc_movement(flow):
        try:
            meanX, meanY, meanZ = np.mean(flow,axis=1)
        except ZeroDivisionError:
            meanX, meanY, meanZ = 0, 0, 0
        movement = np.array([meanX, meanY, meanZ])
        return movement

    def calc_flowAngleVar(flow):
        angle_arr = np.arctan2(flow[0], flow[1]) * 180 / np.pi
        angleVar = np.var(angle_arr)
        return angleVar

    fixDirection_arr = np.zeros((PAGE_MAX + 1,TIME_MAX + 1, 3))  # +1 to adjust to 1 origin of time
    allPage_fixDirection_arr = np.zeros((TIME_MAX + 1, 3))
    latestMovement = np.array([0, 0, 0])
    angleThresh = 6000

    feature_params = dict(maxCorners = 200,
                            qualityLevel = 0.001,
                            minDistance = 10,
                            blockSize = 5)

    for page in range(1, PAGE_MAX + 1):
        prevImg = ciputil.get_image(time=1, page=page)
        prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
        prevFeature = None
        for time in range(2, TIME_MAX + 1):
            if prevFeature is None:
                prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
                prevFeature = cv2.goodFeaturesToTrack(prevGray, mask=None, **feature_params)
            nextImg = ciputil.get_image(time=time, page=page)
            try:
                prevGood, nextGood = get_feature(prevImg, nextImg, prevFeature)
                if prevGood.shape[0] == 0:
                    raise FeatureError("Not detect feature")
                flow = calc_flow(prevGood, nextGood)
                prevFeature = nextGood.reshape(-1, 1, 2)
                movement = calc_movement(flow)
                latestMovement = movement
                angleVar = calc_flowAngleVar(flow)
                if angleVar >= angleThresh:
                    movement = fixDirection_arr[page - 1][time]
                else:
                    pass
            except FeatureError:
                prevFeature = None
                movement = latestMovement
            for i in range(3):
                fixDirection_arr[page][time][i] = fixDirection_arr[page][time - 1][i] + movement[i]
            prevImg = nextImg

    return fixDirection_arr

def output_stabilized_video(fixDirection_arr, configFilepath):
    """
    output stabilized video according to fixDirection_arr
    """

    config = ConfigParser()
    config.read(configFilepath)
    videoFilepath = config["VIDEO"]["VIDEO_FILEPATH"]
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
            fixImg = np.zeros((960, 960, 3), np.uint8)
            height, width = img.shape[:2]
            fixHeight, fixWidth = fixImg.shape[:2]

            fixDirectionX = int(fixDirection_arr[page][time][0])
            fixDirectionY = int(fixDirection_arr[page][time][1])
            fixDirectionZ = int(fixDirection_arr[page][time][2])

            fixImg[int((fixHeight - height) / 2) - fixDirectionY:
                   int(height + (fixHeight - height) / 2) - fixDirectionY,
                   int((fixWidth - width) / 2) - fixDirectionX:
                   int(width + (fixWidth - width) / 2) - fixDirectionX] = img

            text = '[{0:03d},{1:03d},{2:03d}]'.format(fixDirectionX, fixDirectionY,fixDirectionZ)
            cv2.putText(fixImg, text, (fixWidth - 300, fixHeight - 25),
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
    fixDirection_arr = calc_fix_direction()
    np.save("./fixDir.npy", fixDirection_arr)
    print("DONE:  calcurate fix direction")

    if OUTPUT_VIDEO:
        output_stabilized_video(fixDirection_arr, configFilepath)


if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print('\nelapse time: {} sec'.format(elapse))
