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
        flow_arr[time] = flow
        prevStabImg = nextStabImg
    return flow_arr

def calc_cumulative_flows(flow_arr, windowSize, fixDirection_arr):
    """
    return cmlFlow_arr[time+1][960][960][2], 1 origin time
    """
    def make_mask(fixDirection_arr):
        """
        return mask_arr[time+1][960][960][2], 1 origin time
        """
        mask_arr = np.zeros((TIME_MAX + 1, 960, 960, 2))

        for time in range(TIME_MAX + 1):
            fixDirectionX = int(fixDirection_arr[PAGE][time][0])
            fixDirectionY = int(fixDirection_arr[PAGE][time][1])
            mask_arr[time][300 - fixDirectionY : 660 - fixDirectionY, 300 - fixDirectionX : 660 - fixDirectionX] = [1,1]
        return mask_arr

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

    mask_arr = make_mask(fixDirection_arr)
    cmlFlow_arr = np.zeros((TIME_MAX + 1, 960, 960, 2))
    for time in range(2, TIME_MAX + 1):
        print("time:{}".format(time))
        flowStart = time
        flowEnd = min(time + windowSize, TIME_MAX + 1)
        mask = np.prod(mask_arr[flowStart : flowEnd], axis=0)
        cmlFlow_arr[time] = cumulate(flow_arr[flowStart : flowEnd]) * mask
    return cmlFlow_arr

def calc_cumulative_flows_fast(flow_arr, windowSize):
    """
    return cmlFlow_arr[time+1][960][960][2], 1 origin time
    """
    
    def cumulate(flow_arr, mask):
        """
        calculate only if mask pixcel == True, otherwise return (0,0)
        """
        assert mask.shape == (960, 960)
        
        cmlFlow = np.zeros((960, 960, 2))
        for x in range(960):
            for y in range(960):
                if mask[x][y]:
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
    needCalc = np.ones((960,960)).astype(bool)
    cmlFlow_arr[2] = cumulate(flow_arr[2 : min(2+windowSize, TIME_MAX + 1)], needCalc) #initialization with time=2
    
    for time in range(2, TIME_MAX):
        print("calc time:{} from time:{}".format(time+1, time))
        a = cmlFlow_arr[time]-flow_arr[time] #subtraction
        assert a.shape == (960, 960, 2)
        needCalc= np.ones((960, 960)).astype(bool)
        for x in range(960):
            for y in range(960):
                """
                time: (x, y) -> time+1: (i, j) -> time+windowSize: (m, n)
                """
                i = x + int(flow_arr[time][x][y][0])
                j = y + int(flow_arr[time][x][y][1])
                m = i + int(a[x][y][0])
                n = j + int(a[x][y][1])
               
                cmlFlow_arr[time+1][i][j] = a[x][y] #adjust to new coordinate
                if (time+windowSize <= TIME_MAX):
                    cmlFlow_arr[time+1][i][j] += flow_arr[time+windowSize][m][n] #addtion
                needCalc[i][j]=False
        print("\tcalculating {} pixcels".format(np.sum(needCalc)))
        cmlFlow_arr[time+1]+=cumulate(flow_arr[time+1:min(time+1+windowSize, TIME_MAX+1)], needCalc)
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

    _, fixDirectionFilepath, _ = ciputil.read_config_stabilize(configFilepath)
    fixDirection_arr=np.load(fixDirectionFilepath)

    print("START: calculate stabilized dense flows")
    flow_arr = calc_stabilized_flows(fixDirection_arr)

    print("START: cumulating flows, windowSize = {}".format(windowSize))
    #cmlFlow_arr = calc_cumulative_flows(flow_arr, windowSize, fixDirection_arr)
    cmlFlow_arr = calc_cumulative_flows_fast(flow_arr, windowSize)
    
    np.save(dumpFilepath, cmlFlow_arr)
    print("DONE: dump to {}".format(dumpFilepath))

    if outputVideo:
        print("START: output video to {}".format(videoFilepath))
        output_cumulative_video(cmlFlow_arr, fixDirection_arr, videoFilepath)

if __name__=="__main__":
    main()
