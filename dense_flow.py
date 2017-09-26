#! /usr/bin/env python
# coding: utf-8

import sys
import time
import numpy as np
import cv2
from configparser import ConfigParser
import ciputil

TIME_MAX = None
PAGE_MAX = None
OUTPUT_VIDEO = None


def draw_flow(img, flow, step=2):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    for (x1, y1), (x2, y2) in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img

def output_video_with_flow(configFilepath):
    #-----------------------------------------
    # pre processing
    #------------------------------------------
    config = ciputil.set_config(configFilepath)
    videoFilepath = config["VIDEO"]["OPT_VIDEO_FILEPATH"]
    if config["VIDEO"]["PAGE"] == "ALL":
        pageFirst = 1
        pageLast = PAGE_MAX
    else:
        pageFirst = int(config["VIDEO"]["PAGE"])
        pageLast = pageFirst
    print("START: output video to {}, pageFirst: {}, pageLast: {}".format(videoFilepath, pageFirst, pageLast))

    cap = cv2.VideoCapture("./out/output_mode.mp4")
    fourcc = int(cv2.VideoWriter_fourcc(*"avc1"))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(videoFilepath, fourcc, 5.0, (960, 960))
    ret, prev = cap.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            nextGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevGray = nextGray
            flow_img = draw_flow(img, flow, 4)
            video.write(flow_img)
            if cv2.waitKey(1)&0xff == 27:
                break
        else:
            break
    video.release()
    print("DONE: output video to {}".format(videoFilepath))


def main():
    global TIME_MAX
    global PAGE_MAX
    global OUTPUT_VIDEO

    configFilepath = "./config/config.ini"
    TIME_MAX, PAGE_MAX, OUTPUT_VIDEO = ciputil.read_config(configFilepath)

    if OUTPUT_VIDEO:
        output_video_with_flow(configFilepath)

if __name__ == "__main__":
    start = time.time()
    main()
    elapse = time.time() - start
    print("\nelapse time: {} sec".format(elapse))
