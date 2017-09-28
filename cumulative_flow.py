#! /usr/bin/env python
# coding: utf-8

"""
Calc cumulative flow (cmlFlow_arr)　for a certain page.

Please configure in [CUMULATIVE] section of config file.
    PAGE :                page to be calculate cumulative flow.
    WINDOW_SIZE :    The degree of cumulation. When windowSize = 1, cumulative flow equals to usual dence flow.
    DUMP_FILEPATH :  filepath to dump cmlFlow_arr
    VIDEO_FILEPATH : filepath to output video. only used when  DEFAULT.OUTPUT_VIDEO = yes
"""

import numpy as np
import sys
from matplotlib import pyplot as plt
import cv2

import ciputil

TIME_MAX = None
PAGE = None

def calc_stabilized_flows(fixDirection_arr):
    """
    return flow_arr[time+1][960][960][2], 1 origin time
    """
    flow_arr = np.zeros((TIME_MAX + 1, 960, 960, 2))#1 origin flow_arr[time] means change between (time - 1, time)

    prevImg = ciputil.get_image(time=1, page=PAGE)
    prevStabImg = ciputil.get_stabilized_image(prevImg, fixDirection_arr[PAGE][1])

    for time in range(2, TIME_MAX + 1):
        print("time:{}".format(time))
        nextImg = ciputil.get_image(time=time, page=PAGE)
        nextStabImg = ciputil.get_stabilized_image(nextImg, fixDirection_arr[PAGE][time])
        flow = ciputil.calc_dense_flow(prevStabImg, nextStabImg)
        
        prevMask = np.zeros((960, 960, 2))
        prevFixDirectionY = int(fixDirection_arr[PAGE][time - 1][0])
        prevFixDirectionX = int(fixDirection_arr[PAGE][time - 1][1])
        prevMask[240 - prevFixDirectionX : 720 - prevFixDirectionX, 240 - prevFixDirectionY : 720 - prevFixDirectionY] = [1,1]
        nextMask = np.zeros((960, 960, 2))
        nextFixDirectionY = int(fixDirection_arr[PAGE][time][0])
        nextFixDirectionX = int(fixDirection_arr[PAGE][time][1])
        nextMask[240 - nextFixDirectionX : 720 - nextFixDirectionX, 240 - nextFixDirectionY : 720 - nextFixDirectionY] = [1,1]
        mask = prevMask * nextMask

        maskedFlow = flow * mask
        flow_arr[time] = maskedFlow
        prevStabImg = nextStabImg   
    return flow_arr

def calc_cumulative_flows(flow_arr, windowSize):
    """
    return cmlFlow_arr[time+1][960][960][2], 1 origin time
    """
    def cumulate(flow_arr):
        cmlFlow = np.zeros((960, 960, 2))
        for x in range(960):
            for y in range(960):
                i = x
                j = y
                for flow in flow_arr:
                    assert flow.shape == (960, 960, 2)
                    ni = i + flow[int(i)][int(j)][0]
                    nj = j + flow[int(i)][int(j)][1]
                    i, j = ni, nj
                cmlFlow[x][y][0] = i - x
                cmlFlow[x][y][1] = j - y
        return cmlFlow
    
    assert flow_arr.shape == (TIME_MAX + 1, 960, 960, 2)
    assert np.array_equal(flow_arr[0], np.zeros((960, 960, 2)))
    assert np.array_equal(flow_arr[1], np.zeros((960, 960, 2)))
    
    cmlFlow_arr = np.zeros((TIME_MAX + 1, 960, 960, 2))
    for time in range(2, TIME_MAX + 1):
        print("time:{}".format(time))
        flowStart = time
        flowEnd = min(time + windowSize, TIME_MAX + 1) 
        cmlFlow_arr[time] = cumulate(flow_arr[flowStart : flowEnd])
    return cmlFlow_arr
    
def output_cumulative_video(cmlFlow_arr, fixDirection_arr, videoFilepath):
    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    video = cv2.VideoWriter(videoFilepath, fourcc, 5.0, (960, 960))
    for time in range(1, TIME_MAX + 1):
        print("time:{}".format(time))
        img = ciputil.get_image(time=time, page=PAGE)
        stabImg = ciputil.get_stabilized_image(img, fixDirection_arr[PAGE][time])
        if time < TIME_MAX:
            flowImg = ciputil.draw_dense_flow(stabImg, cmlFlow_arr[time])
        else: 
            flowImg = stabImg
        video.write(flowImg)
    video.release()
    
def main():
    global TIME_MAX
    global PAGE
    
    configFilepath = "./config/config.ini"
    TIME_MAX, _, outputVideo = ciputil.read_config(configFilepath)
    PAGE, windowSize, dumpFilepath, videoFilepath = ciputil.read_config_cumulative(configFilepath)

    fixDirectionFilepath = ciputil.read_config_fixDirection(configFilepath)
    fixDirection_arr=np.load(fixDirectionFilepath)
    
    print("START: calculate stabilized dense flows")
    flow_arr = calc_stabilized_flows(fixDirection_arr)
    
    print("START: cumulating flows, windowSize = {}".format(windowSize))
    cmlFlow_arr = calc_cumulative_flows(flow_arr, windowSize = windowSize)
    np.save(dumpFilepath, cmlFlow_arr)
    print("DONE: dump to {}".format(dumpFilepath))
    
    if outputVideo:
        print("START: output video to {}".format(videoFilepath))
        output_cumulative_video(cmlFlow_arr, fixDirection_arr, videoFilepath)   

if __name__=="__main__":
    main()
