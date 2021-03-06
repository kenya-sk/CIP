# coding: utf-8

"""
load configuration data from config/config.ini.
get original/stabilized image.
calculate/draw density flow.
"""

import sys
import cv2
import numpy as np
from configparser import ConfigParser

BASEDIR = None
LEVEL = None

def set_config(configFilepath):
    try:
        config = ConfigParser()
        config.read(configFilepath)
    except FileNotFoundError:
        print("Not Found: {}".format(configFilepath))
        sys.exit(1)
    return config

def read_config(configFilepath, level):
    global BASEDIR
    global LEVEL

    def get_level_configuration():
        #inputFilepath = BASEDIR + "/Pre_Data{0:02d}/input.csv".format(LEVEL)
        inputFilepath = BASEDIR + "/Eva_Data{0:02d}/input.csv".format(LEVEL)
        try:
            with open(inputFilepath,"r") as f:
                f.readline()
                timemax = int(f.readline().strip())
                pagemax = int(f.readline().strip())
        except FileNotFoundError:
            print("Not Found: {}".format(inputFilepath))
            sys.exit(1)
        return timemax, pagemax

    config = set_config(configFilepath)
    BASEDIR = config["DEFAULT"]["BASEDIR"]
    #LEVEL = int(config["DEFAULT"]["LEVEL"])
    LEVEL = level
    output_video = config.getboolean("DEFAULT", "OUTPUT_VIDEO")
    timemax, pagemax = get_level_configuration()

    return timemax, pagemax, output_video


def get_image(time, page):
    filepath = BASEDIR + "/Pre_Data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(LEVEL, time, page)
    #filepath = BASEDIR + "/Eva_Data{0:02d}/t{1:03d}/Eva_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(LEVEL, time, page)
    img = cv2.imread(filepath)
    assert img.shape == (480, 480, 3)
    return img

def calc_dense_flow(prevImg, nextImg):
    if len(prevImg.shape) == 3:
        prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
    if len(nextImg.shape) == 3:
        nextImg =  cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(prevImg, nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def get_stabilized_image(img, fixDirection):
    assert img.shape == (480, 480, 3)
    assert len(fixDirection) == 3 # x,y,z

    fixImg = np.zeros((960, 960, 3), np.uint8)
    height, width = img.shape[:2]
    fixHeight, fixWidth = fixImg.shape[:2]
    #rounding of image movement by image size
    fixDirection[fixDirection > 240] = 240
    fixDirection[fixDirection < -240] = -240

    fixDirectionX = int(fixDirection[0])
    fixDirectionY = int(fixDirection[1])
    fixDirectionZ = int(fixDirection[2])

    fixImg[int((fixHeight - height) / 2) - fixDirectionY:
           int(height + (fixHeight - height) / 2) - fixDirectionY,
           int((fixWidth - width) / 2) - fixDirectionX:
           int(width + (fixWidth - width) / 2) - fixDirectionX] = img

    text = '[{0:03d},{1:03d},{2:03d}]'.format(fixDirectionX, fixDirectionY,fixDirectionZ)
    cv2.putText(fixImg, text, (660, 935),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return fixImg

def draw_dense_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    for (x1, y1), (x2, y2) in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img

def read_config_cumulative(configFilepath):
    config = set_config(configFilepath)
    page = int(config["CUMULATIVE"]["PAGE"])
    windowSize = int(config["CUMULATIVE"]["WINDOW_SIZE"])
    dumpFilepath = config["CUMULATIVE"]["DUMP_FILEPATH"]
    videoFilepath = config["CUMULATIVE"]["VIDEO_FILEPATH"]

    return  page, windowSize, dumpFilepath, videoFilepath

def read_config_stabilize(configFilepath):
    config = set_config(configFilepath)
    angleThresh = float(config["STABILIZE"]["ANGLE_THRESH"])
    dumpFilepath = config["STABILIZE"]["DUMP_FILEPATH"]
    videoFilepath = config["STABILIZE"]["VIDEO_FILEPATH"]

    return angleThresh, dumpFilepath, videoFilepath

def read_config_dot(configFilepath):
    config = set_config(configFilepath)
    recalculate = config.getboolean("DOT", "RECALCULATE")
    windowSize = int(config["DOT"]["WINDOW_SIZE"])
    flowThreshold = int(config["DOT"]["FLOW_THRESHOLD"])
    dotThreshold = int(config["DOT"]["DOT_THRESHOLD"])
    dumpFilepath = config["DOT"]["DUMP_FILEPATH"]
    videoFilepath = config["DOT"]["VIDEO_FILEPATH"]

    return recalculate, windowSize, flowThreshold, dotThreshold, dumpFilepath, videoFilepath
