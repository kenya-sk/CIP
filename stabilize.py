#! /usr/bin/env python
# coding: utf-8

import sys
import time
import numpy as np
import statistics as st
import cv2
import matplotlib.pyplot as plt
from configparser import ConfigParser

BASEDIR = None
LEVEL = None
TIME_MAX = None
PAGE_MAX = None
OUTPUT_VIDEO = None


def set_config(configFilepath):
    global BASEDIR
    global LEVEL
    global TIME_MAX
    global PAGE_MAX
    global OUTPUT_VIDEO

    def get_level_configuration(level):
        '''
        load input.csv for given level
        return timeMax and pageMax
        '''

        inputFilepath = BASEDIR + '/Pre_Data{0:02d}/input.csv'.format(level)
        try:
            with open(inputFilepath, 'r') as f:
                f.readline()
                timeMax = int(f.readline().strip())
                pageMax = int(f.readline().strip())
        except FileNotFoundError:
            print('Not Found: {}'.format(inputFilepath))
            sys.exit(1)
        return timeMax, pageMax

    try:
        config = ConfigParser()
        config.read(configFilepath)
    except FileNotFoundError:
        print('Not Found: {}'.format(configFilepath))
        sys.exit(1)

    BASEDIR = config["DEFAULT"]["BASEDIR"]
    LEVEL = int(config["DEFAULT"]["LEVEL"])
    TIME_MAX, PAGE_MAX = get_level_configuration(LEVEL)
    OUTPUT_VIDEO = config.getboolean('DEFAULT', 'OUTPUT_VIDEO')


def get_image(level, time, page):
    filepath = BASEDIR + "/Pre_Data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(level, time, page)
    img = cv2.imread(filepath)
    assert img.shape == (480, 480, 3)
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

    def calc_movement(flow):
        """
        calcurate movement according to flow
        """
        movement = np.array([1, 1, 0])  # WRITE ME
        return movement

    fixDirection_arr = np.zeros((TIME_MAX + 1, 3))  # +1 to adjust to 1 origin of time
    prevImg = get_image(level=LEVEL, time=1, page=1)
    for time in range(2, TIME_MAX + 1):
        nextImg = get_image(level=LEVEL, time=time, page=1)
        flow = calc_flow(prevImg, nextImg)
        movement = calc_movement(flow)
        for i in range(3):
            fixDirection_arr[time][i] = fixDirection_arr[time - 1][i] + movement[i]
        prevImg = nextImg

    return fixDirection_arr


def output_fix_direction(fixDirection_arr, outputFilepath):
    """
    output fixDirection_arr into text file.
    """
    pass  # WRITE ME


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
            img = get_image(level=LEVEL, time=time, page=page)
            fixImg = np.zeros((960, 960, 3), np.uint8)
            height, width = img.shape[:2]
            fixHeight, fixWidth = fixImg.shape[:2]

            fixDirectionX = int(fixDirection_arr[time][0])
            fixDirectionY = int(fixDirection_arr[time][1])

            fixImg[int((fixHeight - height) / 2) - fixDirectionY:
                   int(height + (fixHeight - height) / 2) - fixDirectionY,
                   int((fixWidth - width) / 2) - fixDirectionX:
                   int(width + (fixWidth - width) / 2) - fixDirectionX] = img

            text = '[{0:03d},{1:03d}]'.format(fixDirectionX, fixDirectionY)
            cv2.putText(fixImg, text, (fixWidth - 200, fixHeight - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            video.write(fixImg)

        for _ in range(10):
            video.write(waitImg)
    video.release()
    print("DONE: output video to {}".format(videoFilepath))


def main():
    configFilepath = "./config/config.ini"
    set_config(configFilepath)
    fixDirection_arr = calc_fix_direction()
    print("DONE:  calcurate fix direction")

    if OUTPUT_VIDEO:
        output_stabilized_video(fixDirection_arr, configFilepath)


if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print('\nelapse time: {} sec'.format(elapse))
