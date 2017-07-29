#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import cv2
import argparse

def main(videoFilepath, level=1,axis='time'):
	try:
		with open("config.txt", 'r') as f:
			direc=f.readline().strip()
	except FileNotFoundError:
		print("NotFound: config.txt")
		print("\tPlease specify file directory in ./config.txt")
		sys.exit(1)
	
	direc+="/Pre_Data{0:02d}".format(level)
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
	if axis == 'time':
		time_scale(timeMax,pageMax,level,direc,video,waitImg)
	else:
		depth_scale(timeMax,pageMax,level,direc,video,waitImg)


def depth_scale(timeMax,pageMax,level,direc,video,waitImg):
	for time in range(1,timeMax+1):
		for page in range(1,pageMax+1):
			filepath=direc+"/t{0:03d}/Pre_Data{1:02d}_t{0:03d}_page_{2:04d}.tif".format(time, level, page)
			img = cv2.imread(filepath)
			video.write(img)
		for _ in range(10):
			video.write(waitImg)
	video.release()


def time_scale(timeMax,pageMax,level,direc,video,waitImg):
	for page in range(1,pageMax+1):
		for time in range(1,timeMax+1):
			filepath=direc+"/t{0:03d}/Pre_Data{1:02d}_t{0:03d}_page_{2:04d}.tif".format(time, level, page)
			img = cv2.imread(filepath)
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
