#! /usr/bin/env python
#coding: utf-8

"""
output answer file(.csv). using DBSCAN clustering algorithm.
"""

import numpy as np
import ciputil
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import time

from sklearn.cluster import DBSCAN

TIME_MAX=None
PAGE_MAX=None

def get_df(dotThreshold, level):
    print("START: load dot")

    dct_lst=[]
    for page in range(1,PAGE_MAX+1):
        dctPage_lst=[]
        dotFilepath="./out/{}_dot_{}.npy".format(level, page)
        #dotFilepath="./out/dot_{}.npy".format(page)
        dotProduct_arr = np.load(dotFilepath)
        for time in range(1, TIME_MAX+1):
            divisionPoint_arr = np.array(np.where(dotProduct_arr[time] < dotThreshold)).T.reshape(-1,2)
            #print("loading {} points".format(divisionPoint_arr.shape[0]))
            for x, y in divisionPoint_arr:
                dct={}
                dct["x"]=x
                dct["y"]=y
                dct["time"]=time
                dct["page"]=page
                dctPage_lst.append(dct)
        if len(dctPage_lst) > TIME_MAX*250:
            dctPage_lst = list(np.random.choice(dctPage_lst, TIME_MAX*250))
        dct_lst.extend(dctPage_lst)
    df=pd.DataFrame(dct_lst)
    return df

def classify(df):
    print("START: classification")

    norm_df=pd.DataFrame(index=df.index)
    norm_df["time"]=df["time"]/400
    norm_df["page"]=df["page"]/260
    norm_df["x"]=df["x"]/960
    norm_df["y"]=df["y"]/960

    dbscan = DBSCAN(eps=0.02,min_samples=100).fit(norm_df)
    df["label"]=dbscan.labels_
    unique, counts = np.unique(df["label"], return_counts=True)
    print(dict(zip(unique, counts)))
    return df

def output(df, fixDirection_arr, outputFilepath):
    numDetect=np.max(df["label"])+1
    cellHeight=8
    cellWidth=50

    with open(outputFilepath, "w") as f:
        divisionLabel_lst=[]
        for label in range(numDetect):
            filter_df=df[df["label"]==label]
            tf=np.min(filter_df["time"]) #time first
            tl=np.max(filter_df["time"])
            if tl - tf > 10:
                divisionLabel_lst.append(label)

        f.write("{}\n".format(TIME_MAX))
        f.write("{}\n".format(len(divisionLabel_lst)))
        f.write("\n")

        for label in divisionLabel_lst:
            filter_df=df[df["label"]==label]
            mx=np.mean(filter_df["x"])
            my=np.mean(filter_df["y"])
            mz=np.mean(filter_df["page"])
            tf=np.min(filter_df["time"]) #time first
            tl=np.max(filter_df["time"])

            for time in range(1, TIME_MAX+1):
                if tf<=time<tl:
                    ax=max(0, int(my-cellWidth/2+fixDirection_arr[int(mz)][time][0]-240))
                    ay=max(0, int(mx-cellWidth/2+fixDirection_arr[int(mz)][time][1]-240))
                    az=max(0, int(mz - cellHeight/2))
                    bx=min(480, int(my+cellWidth/2+fixDirection_arr[int(mz)][time][0]-240))
                    by=min(480, int(mx+cellWidth/2+fixDirection_arr[int(mz)][time][1]-240))
                    bz=min(PAGE_MAX, int(mz + cellHeight/2))
                else:
                    ax,ay,az,bx,by,bz=-1,-1,-1,-1,-1,-1
                f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(ax,ay,az,bx,by,bz))
            f.write("\n")

def main(level):
    global TIME_MAX
    global PAGE_MAX

    configFilepath = "./config/config.ini"
    TIME_MAX, PAGE_MAX, OUTPUT_VIDEO = ciputil.read_config(configFilepath,level)
    #_, _, _ , dotThreshold, _, _ = ciputil.read_config_dot(configFilepath)
    dotThreshold=-10
    df=get_df(dotThreshold, level)
    df=df[::10]
    df=classify(df)

    _, fixDirectionFilepath, _ = ciputil.read_config_stabilize(configFilepath)
    fixDirectionFilepath = fixDirectionFilepath.replace("fixDir.npy", str(level) + "_fixDir.npy")
    print(fixDirectionFilepath)
    fixDirection_arr = np.load(fixDirectionFilepath)

    output(df, fixDirection_arr, "./out/output{}.csv".format(level))

if __name__=="__main__":
    start = time.time()
    main(1)
    elapse = time.time() - start
    print('\nelapse time: {}  sec'.format(elapse))
