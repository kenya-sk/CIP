#! /usr/bin/env python
# coding: utf-8

import sys
import cv2
from configparser import ConfigParser

def read_config(configFilepath):
    try:
        config = ConfigParser()
        config.read(configFilepath)
    except FileNotFoundError:
        print("Not Found: {}".format(configFilepath))
        sys.exit(1)
    
    basedir = config["DEFAULT"]["BASEDIR"]
    level = int(config["DEFAULT"]["LEVEL"])
    output_video = config.getboolean("DEFAULT", "OUTPUT_VIDEO")

    return basedir, level, output_video


def get_image(filepath):
    img = cv2.imread(filepath)
    assert img.shape == (480, 480, 3)
    return img
