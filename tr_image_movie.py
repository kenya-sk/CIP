#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import cv2
import argparse


def get_lst_by_time(direc, level, timeMax, pageMax):
	filepath_lst=[]
	for page in range(1,pageMax+1):
		for time in range(1,timeMax+1):
			filepath=direc+"/Pre_Data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(level, time, page)
			filepath_lst.append(filepath)
		for _ in range(10):
			filepath_lst.append(None)
	return filepath_lst

def get_lst_by_page(direc, level, timeMax, pageMax):
	filepath_lst=[]
	for time in range(1,timeMax+1):
		for page in range(1,pageMax+1):
			filepath=direc+"/Pre_Data{0:02d}/t{1:03d}/Pre_Data{0:02d}_t{1:03d}_page_{2:04d}.tif".format(level, time, page)
			filepath_lst.append(filepath)
		for _ in range(10):
			filepath_lst.append(None)
	return filepath_lst

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
	# video output configuration
	#--------------------------------------------------------------------------------
	fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
	video = cv2.VideoWriter(videoFilepath,fourcc,10.0,(480,480))
	waitImg = np.zeros((480,480,3),np.uint8)#image for waiting


	#--------------------------------------------------------------------------------
	# output .mp4 with designated order 
	#--------------------------------------------------------------------------------
	if axis=="time":
		filepath_lst=get_lst_by_time(direc, level, timeMax, pageMax)
	elif axis=="page":
		filepath_lst=get_lst_by_page(direc, level, timeMax, pageMax)
	else:
		print("Bad Axis Error: {}".format(axis))
		sys.exit(1)

	print("level: {}, axis: {}".format(level, axis))
	for i,filepath in enumerate(filepath_lst):
		if i % 100==0:
			print('.',end='', flush=True)
		if filepath is not None:
			img=cv2.imread(filepath)
			video.write(img)
		else:
			video.write(waitImg)
	print()
	print("DONE: {}".format(videoFilepath))

	
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
