import dlib
import numpy as np
from PIL import Image
import cv2
import os
import csv
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json

with open('data/analysis.json') as json_file:
	face_analyis = json.load(json_file)


def crop_image(img, crop_image_path, left, top, right, bottom):
	faceimg = img[top:bottom, left:right]
	lastimg = cv2.resize(faceimg, (200, 200))
	cv2.imwrite(crop_image_path, lastimg)


def shape_to_numpy_array(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	all_face = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		all_face[i] = (shape.part(i).x, shape.part(i).y)

	return all_face


class CNN(nn.Module):
	def __init__(self,k):
		super(CNN, self).__init__()

		# define the layers
		# kernel size = 3 means (3,3) kernel
		# rgb -> 3 -> in channel
		# number of feature maps = 16
		# number of filters = 3 x 16
		self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		# MaxPool2d, AvgPool2d.
		# The first 2 = 2x2 kernel size,
		# The second 2 means the stride=2

		self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)

		# FC layer
		self.fc1 = nn.Linear(32 * 48 * 48, k) # NUM OF CLUSTER
	
	def forward(self, x):
		# define the data flow through the deep learning layers
		x = self.pool(F.relu(self.l1(x))) # 16x16 x 14 x 14
		#print(x.shape)
		x = self.pool(F.relu(self.l2(x))) # 16x32x6x6
		#print(x.shape)
		x = x.reshape(-1, 32*48*48) # [16 x 1152]# CRUCIAL:
		#print(x.shape)
		x = self.fc1(x)
		return x


def description(region_name,region_type):
	for region in face_analyis["face_regions"]:
		if region["name"] == region_name:
			for feature in region["features"]:
				if feature["name"] == region_type:
					return feature


def fortuneTelling(image_path):
	# Path to images
	image_name = os.path.basename(image_path)
	crop_image_path = os.path.join(os.path.dirname(image_path), "crop_" + image_name)

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	if len(rects) == 1:
		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the landmark (x, y)-coordinates to a NumPy array
			shape = predictor(gray, rect)
			all_face = shape_to_numpy_array(shape)
			left = rect.left()
			top = rect.top()
			right = rect.right()
			bottom = rect.bottom()
			crop_image(image, crop_image_path, left, top, right, bottom)

		eye_model = CNN(3)
		eye_model.load_state_dict(torch.load("models/eye_model.pt"))
		# eye_model.eval()
		eyebrow_model = CNN(3)
		eyebrow_model.load_state_dict(torch.load("models/eyebrow_model.pt"))
		jaw_model = CNN(4)
		jaw_model.load_state_dict(torch.load("models/jaw_model.pt"))
		mouth_model = CNN(3)
		mouth_model.load_state_dict(torch.load("models/mouth_model.pt"))
		nose_model = CNN(3)
		nose_model.load_state_dict(torch.load("models/nose_model.pt"))

		preprocess = torchvision.transforms.Compose([
			# torchvision.transforms.RandomAffine(10),
			torchvision.transforms.ToTensor()
		])

		img = Image.open(crop_image_path)
		img_tensor = preprocess(img)
		img_tensor.unsqueeze_(0)
		eyebrow = eyebrow_model(Variable(img_tensor))
		eye = eye_model(Variable(img_tensor))
		nose = nose_model(Variable(img_tensor))
		mouth = mouth_model(Variable(img_tensor))
		jaw = jaw_model(Variable(img_tensor))

		types = dict()
		types["eyebrow"] = ["Arch","Circle","Straight"]
		types["eye"] = ["Big","Slit","Small"]
		types["nose"] = ["Long","Small","Wide"]
		types["mouth"] = ["Medium","Small","Thick"]
		types["jaw"] = ["Circle","Oval","Square","Triangle"]

		eyebrow_shape = types["eyebrow"][torch.argmax(eyebrow).item()]
		eye_shape = types["eye"][torch.argmax(eye).item()]
		nose_shape = types["nose"][torch.argmax(nose).item()]
		mouth_shape = types["mouth"][torch.argmax(mouth).item()]
		jaw_shape = types["jaw"][torch.argmax(jaw).item()]

		return {
			"image_name": image_name,
			"descriptions": [
				description("eyebrows",eyebrow_shape),
				description("eyes",eye_shape),
				description("nose",nose_shape),
				description("mouth",mouth_shape),
				description("face",jaw_shape),
			]
		}
	elif len(rects) > 1:
		return 1
	else:
		return 0


if __name__ == '__main__':
	fortuneTelling("test.jpg")
