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

LEVEL = None
TIME_MAX = None
PAGE_MAX = None
OUTPUT_VIDEO = None


def draw_flow(img, flow):
    """
    draw optical flow on img
    """
    x, y = img.shape[:2]
    fx,fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    col = (0, 0, 255)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(img, (x1, y1), (x2, y2), col, 1)
    return img    


def calc_fix_direction():
    """
    calculate fix direction for at each time point.
    fixDirection_arr[time] = (fixDirectionX, fixDirectionY, fixDirectionZ)
    """

    def calc_flow(prevImg, nextImg):
        assert prevImg.shape == nextImg.shape
        if len(prevImg.shape) == 3:
            prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
            nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(prevImg, nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)


    def calc_movement(flow, criterion="mode"):
        """
        calcurate movement according to flow
        """
        height, width = 480, 480  # input image size 
        step = 16  # sampling step
        y, x = np.mgrid[step/ 2:height:step,step/ 2:width:step].reshape(2,-1).astype(int)
        movementX = []
        movementY = []
        movementZ = [0, 0]
        for y,x in zip(y,x):
            if abs(flow[y, x][0]) >= 10 or abs(flow[y,x][1]) >= 10:
                movementX.append(int(flow[y,x][0]))
                movementY.append(int(flow[y,x][1]))
        if criterion == "mode":
            try:
                modeX = st.mode(movementX)
                modeY = st.mode(movementY)
                modeZ = st.mode(movementZ)
            except st.StatisticsError:
                modeX = 0
                modeY = 0
                modeZ = 0
            movement = np.array([modeX, modeY, modeZ])
        elif criterion == "median":
            try:
                medianX = st.median(movementX)
                medianY = st.median(movementY)
                medianZ = st.median(movementZ)
            except st.StatisticsError:
                medianX = 0
                medianY = 0
                medianZ = 0
            movement = np.array([medianX, medianY, medianZ])
        elif criterion == "mean":
            try:
                meanX = sum(movementX)/len(movementX)
                meanY = sum(movementY)/len(movementY)
                meanZ = sum(movementZ)/len(movementZ)
            except ZeroDivisionError:
                meanX = 0
                meanY = 0
                meanZ = 0
            movement = np.array([meanX, meanY, meanZ])
        return movement

    def calc_mean_movement(flow):
        """
        calcurate mean movement according to flow
        """
        height, width = 480, 480
        step = 16
        y, x = np.mgrid[step/ 2:height:step, step/ 2:width:step].reshape(2,-1).astype(int)
        
        return movement

    fixDirection_arr = np.zeros((TIME_MAX + 1, 3))  # +1 to adjust to 1 origin of time
    prevImg = ciputil.get_image(level=LEVEL, time=1, page=1)
    for time in range(2, TIME_MAX + 1):
        nextImg = ciputil.get_image(level=LEVEL, time=time, page=1)
        flow = calc_flow(prevImg, nextImg)
        movement = calc_movement(flow, "median")
        for i in range(3):
            fixDirection_arr[time][i] = fixDirection_arr[time - 1][i] + movement[i]
        prevImg = nextImg

    return fixDirection_arr


def output_fix_direction(fixDirection_arr, outputFilepath):
    """
    output fixDirection_arr into text file.
    """
    with open (outputFilepath,"w") as f:
        for i,data in enumerate(fixDirection_arr):
            f.write("time: {0:02d}   fixDirection: {1}\n".format(i, data))

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
            img = ciputil.get_image(level=LEVEL, time=time, page=page)
            fixImg = np.zeros((960, 960, 3), np.uint8)
            height, width = img.shape[:2]
            fixHeight, fixWidth = fixImg.shape[:2]

            fixDirectionX = int(fixDirection_arr[time][0])
            fixDirectionY = int(fixDirection_arr[time][1])
            fixDirectionZ = int(fixDirection_arr[time][2])

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
    output_fix_direction(fixDirection_arr, "./fixDirc.txt")
    print("DONE: output video to {}".format(videoFilepath))


def main():
    global LEVEL
    global TIME_MAX
    global PAGE_MAX
    global OUTPUT_VIDEO

    configFilepath = "./config/config.ini"
    LEVEL, TIME_MAX, PAGE_MAX, OUTPUT_VIDEO = ciputil.read_config(configFilepath)
    fixDirection_arr = calc_fix_direction()
    print("DONE:  calcurate fix direction")

    if OUTPUT_VIDEO:
        output_stabilized_video(fixDirection_arr, configFilepath)


if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print('\nelapse time: {} sec'.format(elapse))
