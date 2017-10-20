import numpy as np
import ciputil
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from sklearn.cluster import DBSCAN

TIME_MAX=None
PAGE_MAX=None

def get_df(dotThreshold):
    print("START: load dot")
    
    dct_lst=[]
    for page in range(1,PAGE_MAX+1):
        dctPage_lst=[]
        dotFilepath="./out/dot_{}.npy".format(page)
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
        if len(dctPage_lst) > 10000:
            dctPage_lst = np.random.choice(dctPage_lst, 10000)
        dct_lst.extend(dctPage_lst)
    df=pd.DataFrame(dct_lst)
    return df

def classify(df):
    print("START: classification")
    
    norm_df=pd.DataFrame(index=df.index)
    norm_df["time"]=df["time"]/TIME_MAX/5
    norm_df["page"]=df["page"]/PAGE_MAX/10
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
        divisionEvent_lst=[]
        for label in range(numDetect):
            filter_df=df[df["label"]==label]
            tf=np.min(filter_df["time"]) #time first
            tl=np.max(filter_df["time"])
            if tl - tf > 10:
                divisionEvent_lst.append(label)
        numDetect = len(divisionEvent_lst)
        
        f.write("{}\n".format(TIME_MAX))
        f.write("{}\n".format(numDetect))
        f.write("\n")

        for label in divisionEvent_lst:
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

def main():
    global TIME_MAX
    global PAGE_MAX
    
    configFilepath = "./config/config.ini"
    TIME_MAX, PAGE_MAX, OUTPUT_VIDEO = ciputil.read_config(configFilepath)
    #_, _, _ , dotThreshold, _, _ = ciputil.read_config_dot(configFilepath)
    dotThreshold=-10
    df=get_df(dotThreshold)
    df=df[::10]
    df=classify(df)
    
    _, fixDirectionFilepath, _ = ciputil.read_config_stabilize(configFilepath)
    fixDirection_arr = np.load(fixDirectionFilepath)
    
    output(df, fixDirection_arr, "output.csv")

if __name__=="__main__":
    main()