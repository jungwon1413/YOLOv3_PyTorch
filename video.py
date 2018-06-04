from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from detect import letterbox_image


# detector.py is the file that we will execute to run our detector.

def arg_parse():
	"""
	Parse arguments to the detect module

	"""

	parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

	parser.add_argument("--video", dest = 'video', help = 
						"Video to run detection upon",
						default = "video.avi", type = str)
	parser.add_argument("--dataset", dest = 'dataset', help = 
						"Dataset on which the network has been trained",
						default = "pascal", type = str)
	parser.add_argument("--confidence", dest = "confidence", help = 
						"Object Confidence to filter predictions",
						default = 0.5)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshold", 
						default = 0.4)
	parser.add_argument("--cfg", dest = 'cfgfile', help = 
						"Config file",
						default = "cfg/yolov3.cfg", type = str)
	parser.add_argument("--weights", dest = 'weightsfile', help = 
						"weightsfile",
						default = "yolov3.weights", type = str)
	parser.add_argument("--reso", dest = 'reso', help = 
						"Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
						default = "416", type = str)


	return parser.parse_args()



def prep_image(img, inp_dim):
	"""
	Prepare image for inputting to the neural network.

	Returns a Variable.

	"""
	orig_im = img
	dim = orig_im.shape[1], orig_im.shape[0]
	img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
	img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
	img_ = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

	return img_, orig_im, dim



def write(x, results, colors):
	c1 = tuple(x[1:3].int())
	c2 = tuple(x[3:5].int())
	cls = int(x[-1])
	label = "{0}".format(classes[cls])
	color = random.choice(colors)
	cv2.rectangle(img, c1, c2, color, 1)
	t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	cv2.rectangle(img, c1, c2, color, -1)
	cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

	return img



args = arg_parse()
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0

CUDA = torch.cuda.is_available()

# Load the class file in our program
num_classes = 80	# For COCO
# classes = load_classes("data/coco.names")			in the loop


# Device setting
device = torch.device("cuda" if CUDA else "cpu")

box_attrs = 5 + num_classes

# Initialize the network and load weights
## Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile).to(device)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# Following code is no longer used in PyTorch 0.4.0
"""
# If there's a GPU available, put the model on GPU
if CUDA:
	model.cuda()
"""


# Set the model in evaluation mode
model.eval()

videofile = args.video

cap = cv2.VideoCapture(videofile)
# cap = cv2.VideoCapture(0) for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0
start = time.time()

while cap.isOpened():
	ret, frame = cap.read()

	if ret:
		img, orig_im, dim = prep_image(frame, inp_dim)

		# cv2.imshow("a", frame)
		# im_dim = frame.shape[1], frame.shape[0]
		im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

		im_dim = im_dim.to(device)
		img = img.to(device)

		with torch.no_grad():
			output = model(img, CUDA)		# PyTorch 0.4.0 style
		output = write_results(output, confidence, num_classes, nms_conf=nms_thresh)


		if type(output) == int:
			frames += 1
			print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
			cv2.imshow("frame", orig_im)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				break
			continue



		output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim))

		im_dim = im_dim.repeat(output.size(0), 1) / inp_dim
		output[:, 1:5] *= im_dim

		classes = load_classes('data/coco.names')
		colors = pkl.load(open("pallete", "rb"))

		list(map(lambda x: write(x, orig_im, colors), output))

		cv2.imshow("frame", orig_im)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		frames += 1
		print(time.time() - start)
		print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
	else:
		break
