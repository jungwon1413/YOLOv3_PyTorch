from __future__ import division

from util import *

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EmptyLayer(nn.Module):
	def __init__(self):
		super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors


class Darknet(nn.Module):
	def __init__(self, cfgfile):
		super(Darknet, self).__init__()
		self.blocks = parse_cfg(cfgfile)
		self.net_info, self.module_list = create_modules(self.blocks)

	

	# Implementing the forward pass of the network
	def forward(self, x, CUDA):
		modules = self.blocks[1:]
		outputs = {}	# We cache the outputs for the route layer

		write = 0		# whether we have encountered the first detection or not
		for i, module in enumerate(modules):
			module_type = (module["type"])

			# Convolutional and Upsample Layers
			if module_type == "convolutional" or module_type == "upsample":
				x = self.module_list[i](x)

			# Route Layer
			elif module_type == "route":
				layers = module["layers"]
				layers = [int(a) for a in layers]

				if (layers[0]) > 0:
					layers[0] = layers[0] - i

				if len(layers) == 1:
					x = outputs[i + (layers[0])]

				else:
					if (layers[1]) > 0:
						layers[1] = layers[1] - i

					map1 = outputs[i + layers[0]]
					map2 = outputs[i + layers[1]]

					x = torch.cat((map1, map2), 1)
			
			# Shortcut Layer
			elif module_type == "shortcut":
				from_ = int(module["from"])
				x = outputs[i-1] + outputs[i+from_]

			# Detection Layer Revisited for YOLO layer (with predict_transform)
			elif module_type == "yolo":
				anchors = self.module_list[i][0].anchors

				# Get the input dimensions
				inp_dim = int(self.net_info["height"])

				# Get the number of classes
				num_classes = int(module["classes"])

				# Transform
				x = x.data
				x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
				if not write:			# if no collector has been initialized
					detections = x
					write = 1

				else:
					detections = torch.cat((detections, x), 1)

			outputs[i] = x


		return detections


	def load_weights(self, weightfile):
		
		# Open the weights file
		fp = open(weightfile, "rb")

		# The first 5 values are header information
		# 1. Major version number
		# 2. Minor version number
		# 3. Subversion number
		# 4,5. Images seen by the network (during training)
		header = np.fromfile(fp, dtype=np.int32, count=5)
		self.header = torch.from_numpy(header)		
		self.seen = self.header[3]				# 4th

		# Load rest of the weights in a np.adarray
		weights = np.fromfile(fp, dtype=np.float32)

		# Iterate over the weights file, and load the weights
		ptr = 0			# To keep track of where we are in the weights array
		for i in range(len(self.module_list)):
			module_type = self.blocks[i + 1]["type"]

			# If module_type is convolutional, load weights
			# Otherwise, ignore.
			
			if module_type == "convolutional":
				model = self.module_list[i]
				try:
					batch_normalize = int(self.blocks[i+1]["batch_normalize"])
				except:
					batch_normalize = 0


				conv = model[0]


				# If batch_normalize is True, we load the weights as follows.
				if (batch_normalize):
					bn = model[1]

					# Get the number of weights of Batch Norm Layer
					num_bn_biases = bn.bias.numel()
	
					# Load the weights (in order)
					bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
					ptr += num_bn_biases
	
					bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr += num_bn_biases
	
					bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr += num_bn_biases

					bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr += num_bn_biases

					# Cast the loaded weights into dims of model weights.
					bn_biases = bn_biases.view_as(bn.bias.data)		 # same shape as ...
					bn_weights = bn_weights.view_as(bn.weight.data)
					bn_running_mean = bn_running_mean.view_as(bn.running_mean)
					bn_running_var = bn_running_var.view_as(bn.running_var)

					# Copy the data to model
					# What: Copies the elements from (src) into tensor.
					bn.bias.data.copy_(bn_biases)		# Tensor copy from bn_biases to bn.bias.data
					bn.weight.data.copy_(bn_weights)
					bn.running_mean.copy_(bn_running_mean)
					bn.running_var.copy_(bn_running_var)

				# If batch_norm is not True, simply load the biases of the conv layer
				else:
					# Number of biases
					num_biases = conv.bias.numel()

					# Load the weights
					conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
					ptr = ptr + num_biases

					# reshape the loaded weights according to the dims of the model weights
					conv_biases = conv_biases.view_as(conv.bias.data)

					# Finally copy the data
					conv.bias.data.copy_(conv_biases)



				# Load the conv layer's weights
				num_weights = conv.weight.numel()

				# Do the same as above for weights
				conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
				ptr = ptr + num_weights

				conv_weights = conv_weights.view_as(conv.weight.data)
				conv.weight.data.copy_(conv_weights)



def get_test_input():
	img = cv2.imread("dog-cycle-car.png")
	img = cv2.resize(img, (416, 416))				# Resize to the input dimension
	img_ = img[:, :, :: -1].transpose((2, 0, 1))	# BGR -> RGB | H X W C -> C X H X W
	img_ = img_[np.newaxis, :, :, :] / 255.0		# Add a channel at 0 (for batch) | Normalize
	img_ = torch.from_numpy(img_).float()			# Convert to float
#	img_ = Variable(img_)							# Convert to Variable
# Variable deprecated in PyTorch 0.4.0 (torch.autograd.Variable == torch.Tensor)

	return img_



def parse_cfg(cfgfile):
	"""
	Takes a configuration file

	Returns a list of blocks. Each block describes a block in the neural
	network to be build. Block is represented as a dictionary in the list

	"""

	file = open(cfgfile, 'r')
	lines = file.read().split('\n')					# store the lines in a list
	lines = [x for x in lines if len(x) > 0]	# get read of the empty lines
	lines = [x for x in lines if x[0] != '#']	# get rid of comments
	lines = [x.rstrip().lstrip() for x in lines]		# get rid of fringe whitespaces


	# Loop over the resultant list to get blocks
	block = {}
	blocks = []
	

	for line in lines:
		if line[0] == "[":				# This marks the start of a new block
			if len(block) != 0:			# If block is not empty, implies it is storing values of previous block.
				blocks.append(block)		# add it the blocks list
				block = {}			# re-init the block
			block["type"] = line[1:-1].rstrip()
		else:
			key, value = line.split("=")
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks



def create_modules(blocks):

# nn.ModuleList
	net_info = blocks[0]		# Captures the information about the input and pre-processing
	module_list = nn.ModuleList()
	prev_filters = 3
	output_filters = []

	for index, x in enumerate(blocks[1:]):
		module = nn.Sequential()

		# check the type of block
		# create a new module for the block
		# append to module_list


		if (x["type"] == "convolutional"):
			# Get the info about the layer
			activation = x["activation"]
			try:
				batch_normalize = int(x["batch_normalize"])
				bias = False
			except:
				batch_normalize = 0
				bias = True


			filters = int(x["filters"])
			padding = int(x["pad"])
			kernel_size = int(x["size"])
			stride = int(x["stride"])

			if padding:
				pad = (kernel_size - 1) // 2
			else:
				pad = 0


			# Add the convolutional layer
			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
			module.add_module("conv_{0}".format(index), conv)

			# Add the Batch Norm Layer
			if batch_normalize:
				bn = nn.BatchNorm2d(filters)
				module.add_module("batch_norm_{0}".format(index), bn)

			# Check the activation
			# It is either Linear or a Leaky ReLU for YOLO
			if activation == "leaky":
				activn = nn.LeakyReLU(0.1, inplace=True)
				module.add_module("leaky_{0}".format(index), activn)

		# If it's an upsampling layer
		# We use Bilinear2dUpsampling
		elif (x["type"] == "upsample"):
			stride = int(x["stride"])
			upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
			module.add_module("upsample_{}".format(index), upsample)


# Route Layer / Shortcut Layers

		# If it is a route layer
		elif (x["type"] == "route"):
			x["layers"] = x["layers"].split(',')
		
			# Start of a route
			start = int(x["layers"][0])
	
			# end, if there exists one.
			try:
				end = int(x["layers"][1])
			except:
				end = 0

			# Positive annotation
			if start > 0:
				start = start - index
			if end > 0:
				end = end - index
			
			route = EmptyLayer()
			module.add_module("route_{0}".format(index), route)
			
			if end < 0:
				filters = output_filters[index + start] + output_filters[index + end]
			else:
				filters = output_filters[index + start]


		# shortcut corresponds to skip connection
		elif x["type"] == "shortcut":
			shortcut = EmptyLayer()
			module.add_module("shortcut_{}".format(index), shortcut)



# YOLO Layer
		# Yolo is the detection layer
		elif x["type"] == "yolo":
			mask = x["mask"].split(",")
			mask = [int(x) for x in mask]

			anchors = x["anchors"].split(",")
			anchors = [int(a) for a in anchors]
			anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors)
			module.add_module("Detection_{}".format(index), detection)

		module_list.append(module)
		prev_filters = filters
		output_filters.append(filters)

	return (net_info, module_list)





model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
