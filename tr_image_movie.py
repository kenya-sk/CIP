#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import cv2
import argparse


def add_frame(img, frame):
    assert len(frame)==4
    cv2.rectangle(img, (frame[0], frame[1]), (frame[2], frame[3]), (0, 0, 255), 2)
    return img

def main(videoFilepath, level=1,axis='time'):

    #--------------------------------------------------------------------------------
    # load config.txt
    #--------------------------------------------------------------------------------
    try:
        with open("config.txt", 'r') as f:
            direc=f.readline().strip()
    except FileNotFoundError:
        print("Not Found: config.txt")
        print("\tPlease specify file directory in ./config.txt")
        sys.exit(1)

    #--------------------------------------------------------------------------------
    # load input.csv to get timeMax and pageMax
    #--------------------------------------------------------------------------------
    inputFilepath=direc+"/Pre_Data{0:02d}/input.csv".format(level)
    with open(inputFilepath, 'r') as f:
        f.readline()
        timeMax=int(f.readline().strip())
        pageMax=int(f.readline().strip())


    #--------------------------------------------------------------------------------
    # load Answer.txt to get time2zrange and time2frame
    #--------------------------------------------------------------------------------
    #answerFilepath=direc+"/Pre_Data{0:02d}/Pre_Data{0:02d}_Answer.txt".format(level)
    answerFilepath=direc+"/CIP/output.csv"
    with open(answerFilepath, 'r') as f:
        assert timeMax==int(f.readline().strip())
        numDivision=int(f.readline().strip())
        #assert numDivision==1
        time2zrange_lst=[]
        time2frame_lst=[]
        for _ in range(numDivision):
            time2zrange=[None]
            time2frame=[None]
            f.readline()
            for time in range(1, timeMax+1):
                ax,ay,az,bx,by,bz=[int(i) for i in f.readline().strip().split('\t')]
                time2zrange.append((az, bz))
                time2frame.append((ax,ay,bx,by))
            time2zrange_lst.append(time2zrange)
            time2frame_lst.append(time2frame)


    #--------------------------------------------------------------------------------
    # video output configuration
    #--------------------------------------------------------------------------------
    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    video = cv2.VideoWriter(videoFilepath,fourcc,10.0,(480,480))
    waitImg = np.zeros((480,480,3),np.uint8)#image for waiting


    #--------------------------------------------------------------------------------
    # output .mp4 with designated order
    #--------------------------------------------------------------------------------
    print("level: {}, axis: {}".format(level, axis))
    if axis == "time":
        time_scale(timeMax,pageMax,level,direc,video,waitImg, time2zrange_lst, time2frame_lst)
    elif axis == "page":
        page_scale(timeMax,pageMax,level,direc,video,waitImg, time2zrange_lst, time2frame_lst)
    else:
        print("Bad Axis Error: {}".format(axis))
        sys.exit(1)
    print("DONE: {}".format(videoFilepath))


def time_scale(timeMax,pageMax,level,direc,video,waitImg, time2zrange_lst, time2frame_lst):
    for page in range(1,pageMax+1):
        for time in range(1,timeMax+1):
            filepath=direc+"/Pre_Data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(level, time, page)
            img = cv2.imread(filepath)
            for time2zrange, time2frame in zip(time2zrange_lst, time2frame_lst):
                if time2zrange[time][0]<=page and page<=time2zrange[time][1]:
                    img = add_frame(img, time2frame[time])
            video.write(img)
        for _ in range(10):
            video.write(waitImg)
    video.release()

def page_scale(timeMax,pageMax,level,direc,video,waitImg, time2zrange_lst, time2frame_lst):
    for time in range(1,timeMax+1):
        for page in range(1,pageMax+1):
            filepath=direc+"/Pre_Data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(level, time, page)
            img = cv2.imread(filepath)
            for time2zrange, time2frame in zip(time2zrange_lst, time2frame_lst):
                if time2zrange[time][0]<=page and page<=time2zrange[time][1]:
                    img = add_frame(img, time2frame[time])
            video.write(img)
        for _ in range(10):
            video.write(waitImg)
    video.release()

def make_parse():
    parser = argparse.ArgumentParser(prog='tr_image_movie.py',
                                    usage='Transrate image to movie',
                                    description='description',
                                    epilog='end',
                                    add_help=True,
                                    )

    parser.add_argument('Arg1: output file path',help='string',type=argparse.FileType('w'))
    parser.add_argument('Arg2: select input image level',default=1,help='integer',type=int)
    parser.add_argument('Arg3: select scale depth or time',default='time',help='string')

    args = parser.parse_args()

if __name__=="__main__":
    make_parse()
    args = sys.argv
    main(str(args[1]), int(args[2]),str(args[3]))
