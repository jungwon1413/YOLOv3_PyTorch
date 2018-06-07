### Transforming the output ###

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


"""
'predict_transform' function processes bounding box in different scales
on a single tensor, rather than several tensors.

prediction: output
inp_dim: input image dimension
anchors: anchors
num_classes: number of classes
CUDA: CUDA flag
"""

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
	batch_size = prediction.size(0)
	stride = inp_dim // prediction.size(2)
	grid_size = inp_dim // stride
	# grid_size = int(prediction.size(2))
	bbox_attrs = 5 + num_classes		# 5 + C
	num_anchors = len(anchors)

	prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
	prediction = prediction.transpose(1,2).contiguous()
	prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

	anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

	# Sigmoid the centre_X, centre_Y, and object confidence
	prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])	# center_X
	prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])	# center_Y
	prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])	# object conf score


	# Add the center offsets
	grid = np.arange(grid_size)
	a, b = np.meshgrid(grid, grid)

	x_offset = torch.FloatTensor(a).view(-1, 1)			# Equivalent to tf.reshape
	y_offset = torch.FloatTensor(b).view(-1, 1)

	# JW: MODIFICATION from original source code
	device = torch.device("cuda"if CUDA else "cpu")

	# if torch.cuda.is_available() == True:
	x_offset = x_offset.to(device)
	y_offset = y_offset.to(device)

	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

	prediction[:, :, :2] += x_y_offset


	# log space transform height and the width
	anchors = torch.FloatTensor(anchors)

	# JW: MODIFICATION from original source code
	#if CUDA:
	# if torch.cuda.is_available() == True:
	anchors = anchors.to(device)

	anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
	prediction[:, :, 2 : 4] = torch.exp(prediction[:, :, 2 : 4]) * anchors	# exponentials to box width/height


	# Class scores
	prediction[:, :, 5 : 5 + num_classes] = torch.sigmoid((prediction[:, :, 5 : 5 + num_classes]))


	# Resize the detections map
	prediction[:, :, : 4] *= stride


	return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
	# Outputs a tensor of shape D x 8
	# D: true detections in all images, each represented by a row
	"""
	prediction: prediction
	confidence: object score threshold
	num_classes: number of classes (80, in our case)
	nms_conf: the NMS IoU threshold (Non-Max Supression)
	"""

	# Object Confidence Thresholding
	# (contains information about B x 10,647 bounding boxes)
	conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
	prediction = prediction*conf_mask

	# Performing Non-maximum Supression
	# It's easier to calculate using corner coordinates, 
	# instead of center coordinate, height and width.
	# So we transform (center_x, center_y, height, width)
	# to (top-left_x, top-left_y, right-bottom_x, right-bottom_y)
	box_corner = prediction.new(prediction.shape)
	# center_x - width/2
	box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 3] / 2)
	# center_y - height/2
	box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 2] / 2)
	# center_x + width/2
	box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 3] / 2)
	# center_y + height/2
	box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 2] / 2)
	# Replace to new prediction box
	prediction[:, :, :4] = box_corner[:, :, :4]


	# Confidence thresholding and NMS has to be done for one image at once.
	# Must loop over the first dimension of prediction.
	# (containing indexes of images in a batch)
	batch_size = prediction.size(0)

	write = False

	for ind in range(batch_size):
		image_pred = prediction[ind]		# image Tensor
			# confidence thresholding
			# NMS

		# We're only concerned with the class score having the maximum value.
		max_conf, max_conf_score = torch.max(image_pred[:, 5 : 5 + num_classes], 1)
		max_conf = max_conf.float().unsqueeze(1)
		max_conf_score = max_conf_score.float().unsqueeze(1)
		seq = (image_pred[:, : 5], max_conf, max_conf_score)
		image_pred = torch.cat(seq, 1)


		# Remove object with confidence score less than the threshold
		non_zero_ind = (torch.nonzero(image_pred[:, 4]))
		
		try:
			# image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
			image_pred_ = image_pred[non_zero_ind.squeeze(), :]
		except:
			continue

		# For PyTorch 0.4 compatibility
		# Since the above code with not raise exception for no detection
		# as scalars are supported in PyTorch
		if image_pred_.shape[0] == 0:
			continue

		# Get the various classes detected in the image
		try:
			img_classes = unique(image_pred_[:,-1])		# -1 index holds the class index
		except:
			continue


		for cls in img_classes:

			# get the detections with one particular class
			cls_mask = image_pred_ * (image_pred_[:,-1] == cls).float().unsqueeze(1)
			class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
			# if class_mask_ind.shape[0] == 0:		# Found No objects ( > threshold )
			# 	continue
			image_pred_class = image_pred_[class_mask_ind].view(-1, 7)


			# sort the detections such that the entry with the maximum objectess
			# confidence is at the top
			conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
			image_pred_class = image_pred_class[conf_sort_index]
			idx = image_pred_class.size(0)		# Number of detections


			# Perform NMS
			for i in range(idx):
				# Get the IOUs of all boxes that come after the one we are looking at
				# in the loop
				try:
					ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1 : ])
				except ValueError:
					break

				except IndexError:
					break

				# Zero out all the detections that have IoU > threshold
				iou_mask = (ious < nms_conf).float().unsqueeze(1)
				image_pred_class[i+1 : ] *= iou_mask

				# Remove the non-zero entries
				non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
				image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)


				"""
				Each detection has 8 attributes
				- Index of the image in the batch
				- 4 corner coordinates
				- Objectness score
				- The score of class (with maximum confidence)
				- The index of that class
				"""
				batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
				# Repeat the batch_id for as many detections of the class cls in the image
				seq = batch_ind, image_pred_class

				if not write:
					output = torch.cat(seq, 1)
					write = True
				else:
					out = torch.cat(seq, 1)
					output = torch.cat((output, out))

		try:
			return output
		except:
			return 0




# To get classes present in any given image
def unique(tensor):
	tensor_np = tensor.cpu().numpy()
	unique_np = np.unique(tensor_np)
	unique_tensor = torch.from_numpy(unique_np)

	tensor_res = tensor.new(unique_tensor.shape)
	tensor_res.copy_(unique_tensor)
	
	return tensor_res



# Calculating the IoU
def bbox_iou(box1, box2):
	"""
	Returns the IoU of two bounding boxes

	"""
	# Get the coordinates of bounding boxes
	b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
	b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

	# Get the coordinates of the intersection rectangle
	inter_rect_x1 = torch.max(b1_x1, b2_x1)
	inter_rect_y1 = torch.max(b1_y1, b2_y1)
	inter_rect_x2 = torch.max(b1_x2, b2_x2)
	inter_rect_y2 = torch.max(b1_y2, b2_y2)

	# Intersection area
	inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

	# Union area
	b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
	b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

	iou = inter_area / (b1_area + b2_area - inter_area)

	return iou



def load_classes(namesfile):
	fp = open(namesfile, "r")
	names = fp.read().split("\n")[:-1]

	return names
