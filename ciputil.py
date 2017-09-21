#! /usr/bin/env python
# coding: utf-8

import sys
import cv2
from configparser import ConfigParser

BASEDIR = None
LEVEL = None

def read_config(configFilepath):
    global BASEDIR
    global LEVEL

    def get_level_configuration(level):
        inputFilepath = BASEDIR + "/Pre_Data{0:02d}/input.csv".format(LEVEL)
        try:
            with open(inputFilepath,"r") as f:
                f.readline()
                timemax = int(f.readline().strip())
                pagemax = int(f.readline().strip())
        except FileNotFoundError:
            print("Not Found: {}".format(inputFilepath))
            sys.exit(1)
        return timemax, pagemax

    try:
        config = ConfigParser()
        config.read(configFilepath)
    except FileNotFoundError:
        print("Not Found: {}".format(configFilepath))
        sys.exit(1)

    BASEDIR = config["DEFAULT"]["BASEDIR"]
    LEVEL = int(config["DEFAULT"]["LEVEL"])
    output_video = config.getboolean("DEFAULT", "OUTPUT_VIDEO")
    timemax, pagemax = get_level_configuration(LEVEL)

    return timemax, pagemax, output_video


def get_image(time, page):
    filepath = BASEDIR + "/Pre_data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(LEVEL, time, page)
    img = cv2.imread(filepath)
    assert img.shape == (480, 480, 3)
    return img
