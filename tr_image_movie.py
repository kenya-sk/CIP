#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import cv2

def main(videoFilepath, level=1):
	direc="/data/sakka/pre_image/Pre_Data{0:02d}".format(level)

	inputFilepath=direc+"/input.csv"
	with open(inputFilepath, 'r') as f:
		f.readline()
		timeMax=int(f.readline().strip())
		pageMax=int(f.readline().strip())
	print(timeMax, pageMax)
	print(videoFilepath)

	
	fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
	video = cv2.VideoWriter(videoFilepath,fourcc,10.0,(480,480))
	waitImg = np.zeros((480,480,3),np.uint8)#image for waiting

	for time in range(1,timeMax+1):
		for page in range(1,pageMax+1):
			filepath=direc+"/t{0:03d}/Pre_Data01_t{1:03d}_page_{2:04d}.tif".format(time, time, page)
			img = cv2.imread(filepath)
			video.write(img)
		for k in range(10):
			video.write(waitImg)
	video.release()	

if __name__=="__main__":
	args = sys.argv
	main(str(args[1]), int(args[2]))
