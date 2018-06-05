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



# detector.py is the file that we will execute to run our detector.

def arg_parse():
	"""
	Parse arguments to the detect module

	"""

	parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

	parser.add_argument("--images", dest = 'images', help = 
						"Image / Directory containing images to perform detection upon",
						default = "imgs", type = str)
	parser.add_argument("--det", dest = 'det', help = 
						"Image / Directory to store detections to",
						default = "det", type = str)
	parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
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



def letterbox_image(img, inp_dim):
	'''resize image with unchanged aspect ratio using padding'''
	img_w, img_h = img.shape[1], img.shape[0]
	w, h = inp_dim
	new_w = int(img_w * min(w/img_w, h/img_h))
	new_h = int(img_h * min(w/img_w, h/img_h))
	resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)

	canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)	# padding with color (128, 128, 128)

	canvas[(h-new_h)//2 : (h-new_h)//2+new_h, (w-new_w)//2 : (w-new_w)//2+new_w, :] = resized_image

	return canvas



def prep_image(img, inp_dim):
	"""
	Prepare image for inputting to the neural network.

	Returns a Variable.

	"""

	img = cv2.resize(img, (inp_dim, inp_dim))		# default: 416 x 416
	img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
	img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

	return img



args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

# Load the class file in our program
num_classes = 80	# For COCO
classes = load_classes("data/coco.names")


# Device setting
device = torch.device("cuda" if CUDA else "cpu")


# Initialize the network and load weights
## Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile).to(device)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = int(args.reso)
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


# Read the Input images
# The paths of the image(or images) are stored in a list called 'imlist'
read_dir = time.time()		# a checkpoint used to measure time

# Detection phase
try:
	imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
	imlist = []
	imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
	print("No file or directory with the name {}".format(images))
	exit()


if not os.path.exists(args.det):
	os.makedirs(args.det)

# load images
load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]


# Pytorch Variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

im_dim_list = im_dim_list.to(device)


# Create the Batches
leftover = 0

if (len(im_dim_list) % batch_size):
	leftover = 1

if batch_size != 1:
	num_batches = len(imlist) // batch_size + leftover
	im_batches = [torch.cat((im_batches[i*batch_size : min((i + 1)*batch_size,
						len(im_batches))])) for i in range(num_batches)]


### The Detection Loop ###
write = 0
start_det_loop = time.time()
objs = {}

for i, batch in enumerate(im_batches):
	# load the image
	start = time.time()
	batch = batch.to(device)

	# prediction = model(Variable(batch, volatile = True), CUDA)
	
	with torch.no_grad():
		# prediction = model(Variable(batch), CUDA)
		prediction = model(batch, CUDA)			# PyTorch 0.4.0 Style
	# Update: volatile is deprecated in PyTorch 0.4, and wouldn't be tracked by autograd.
	# (The volatile flag has no effect now.)

	# prediction = prediction[:, scales_indices]

	prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)

	end = time.time()


	if type(prediction) == int:
#		for im_num, image in enumerate(imlist[i*batch_size : min((i + 1)*batch_size, len(imlist))]):
#			im_id = i*batch_size + im_num
#			print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#			print("{0:20s} {1:s}".format("Objects Detected:", ""))
#			print("-" * 60)
		continue


	prediction[:, 0] += i*batch_size	# transform the attribute from index in batch to index in imlist

	if not write:						# If we haven't initialized output
		output = prediction
		write = 1
	else:
		output = torch.cat((output, prediction))

	for im_num, image in enumerate(imlist[i*batch_size : min((i + 1)*batch_size, len(imlist))]):
		im_id = i*batch_size + im_num
		objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
		print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
		print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
		print("-" * 60)
	
	if CUDA:
		torch.cuda.synchronize()	# makes sure that CUDA kernel is synchronized with the CPU



try:
	output
except NameError:
	print("No detections were made")
	exit()


## Drawing bounding boxes on images ##
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

output[:, [1,3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2,4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2


# Undo rescaling
output[:,1:5] /= scaling_factor

# Clip bounding boxes
for i in range(output.shape[0]):
	output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
	output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])


output_recast = time.time()


# Choose random color for bounding boxes from pickle file
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))


# Draw the boxes!
draw = time.time()

# The function draws a rectangle with a color of a random choice.
def write(x, results):
	c1 = tuple(x[1:3].int())
	c2 = tuple(x[3:5].int())
	img = results[int(x[0])]
	cls = int(x[-1])
	label = "{0}".format(classes[cls])
	color = random.choice(colors)
	cv2.rectangle(img, c1, c2, color, 1)		# Create a random colored rectangle
	t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	cv2.rectangle(img, c1, c2, color, -1)		# Create a randomly filled colored rectangle
	cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

	return img


# Draw the bounding boxes on images
list(map(lambda x: write(x, loaded_ims), output))

# Create a list of image addresses to save detections
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

# Write the images with detections to the address
list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()


### Printing Time Summary ###

# To compare how different hyperparameters effect the speed of the detector.
print("SUMMARY")
print("-" * 60)
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Averate time_per_img", (end - load_batch)/len(imlist)))
print("-" * 60)


torch.cuda.empty_cache()
