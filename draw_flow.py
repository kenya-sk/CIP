#! /usr/bin/env python
# coding: utf-8

import sys
import time
import numpy as np
import cv2
from configparser import ConfigParser
import util

BASEDIR = None
LEVEL = None
TIME_MAX = None
PAGE_MAX = None
OUTPUT_VIDEO = None


def get_level_configuration(level):
    inputFilepath = BASEDIR + "/Pre_Data{0:02d}/input.csv".format(level)
    try:
        with open(inputFilepath,"r") as f:
            f.readline()
            timeMax = int(f.readline().strip())
            pageMax = int(f.readline().strip())
    except FileNotFoundError:
        print("Not Found: {}".format(configFilepath))
        sys.exit(1)
    return timeMax, pageMax


def output_video_with_flow(configFilepath):
    #------------------------------------------
    # pre processing 
    #------------------------------------------
    config = ConfigParser()
    config.read(configFilepath)
    videoFilepath = config["VIDEO"]["OPT_VIDEO_FILEPATH"]
    if config["VIDEO"]["PAGE"] == "ALL":
        pageFirst = 1
        pageLast = PAGE_MAX
    else:
        pageFirst = int(config["VIDEO"]["PAGE"])
        pageLast = pageFirst
    print("START: output video to {}, pageFirst: {}, pageLast: {}".format(videoFilepath, pageFirst, pageLast))

    fourcc = int(cv2.VideoWriter_fourcc(*"avc1"))
    video = cv2.VideoWriter(videoFilepath, fourcc, 5.0, (480, 480))
    waitImg = np.zeros((480, 480, 3), np.uint8)

    #--------------------------------------------
    # sparse optical flow parameter
    #-------------------------------------------
    #corner detection parameter of Shi-Tomasi
    feature_params = dict(maxCorners = 200,
                            qualityLevel = 0.001,
                            minDistance = 10,
                            blockSize = 5)
    #parameter of Lucas-Kanade method
    lk_params = dict(winSize = (20, 20),
                        maxLevel = 5,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    for page in range(pageFirst, pageLast + 1):
        prevImg = util.get_image(BASEDIR + \
                        "/Pre_Data{0:02d}/t001/Pre_Data{0:02d}_t001_page_{1:04d}.tif".format(LEVEL, page))
        prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
        prevFeature = cv2.goodFeaturesToTrack(prevGray, mask=None, **feature_params)
        for time in range(1, TIME_MAX + 1):
            filepath = BASEDIR + \
                "/Pre_Data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(LEVEL, time, page)
            mask = np.zeros((480, 480, 3), np.uint8)
            nextImg = util.get_image(filepath)
            nextGray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
            nextFeature, status, err = cv2.calcOpticalFlowPyrLK(prevGray, nextGray, prevFeature, None, **lk_params)
            prevGood = prevFeature[status == 1]
            nextGood = nextFeature[status == 1]

            for i, (nextPoint, prevPoint) in enumerate(zip(nextGood,prevGood)):
                prevX, prevY = prevPoint.ravel()
                nextX, nextY = nextPoint.ravel()
                mask = cv2.line(mask, (nextX, nextY), (prevX, prevY), (0, 0, 255), 2)
                img = cv2.circle(nextImg, (nextX, nextY), 5, (0,0,255), -1)
            img = cv2.add(img, mask)
            video.write(img)
            prevGray = nextGray
            prevFeature = nextGood.reshape(-1, 1, 2)

        for _ in range(10):
            video.write(waitImg)
    video.release()
    print("DONE: output video to {}".format(videoFilepath))


def main():
    global BASEDIR
    global LEVEL
    global TIME_MAX
    global PAGE_MAX
    global OUTPUT_VIDEO

    configFilepath = "./config/config.ini"
    BASEDIR, LEVEL, OUTPUT_VIDEO = util.read_config(configFilepath)
    TIME_MAX, PAGE_MAX = get_level_configuration(LEVEL) 
    
    if OUTPUT_VIDEO:
        output_video_with_flow(configFilepath)

if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print("\nelapse time: {} sec".format(elapse))
