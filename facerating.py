import torch
from PIL import Image
import dlib
import cv2
import os
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_5_face_landmarks.dat')

model_arch = 'resnet18'

if torch.cuda.is_available():
	device = 'cuda:0'
else:
	device = 'cpu'

transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


def load_model(model_arch):
	model = getattr(torchvision.models, model_arch)(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(in_features=num_ftrs, out_features=1, bias=True)
	model = model.to(torch.device(device))
	return model

def shape_to_normal(shape):
	shape_normal = []
	for i in range(0, 5):
		shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
	return shape_normal
	
def get_eyes_nose_dlib(shape):
	nose = shape[4][1]
	left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
	left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
	right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
	right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
	return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

def distance(a, b):
	return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def cosine_formula(length_line1, length_line2, length_line3):
	cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
	return cos_a

def rotate_point(origin, point, angle):
	ox, oy = origin
	px, py = point

	qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
	qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
	return qx, qy


def is_between(point1, point2, point3, extra_point):
	c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
	c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
	c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
	if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
		return True
	else:
		return False


def rotateImage(obj_img, angle=90):
	obj_h, obj_w = obj_img.shape[:2]
	image_center = (int(obj_w / 2), int(obj_h / 2))
	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
	abs_cos = abs(rotation_mat[0, 0])
	abs_sin = abs(rotation_mat[0, 1])
	bound_w = int(obj_h * abs_sin + obj_w * abs_cos)
	bound_h = int(obj_h * abs_cos + obj_w * abs_sin)
	rotation_mat[0, 2] += bound_w / 2 - image_center[0]
	rotation_mat[1, 2] += bound_h / 2 - image_center[1]
	image_rotated = cv2.warpAffine(obj_img, rotation_mat, (bound_w, bound_h))
	return image_rotated, rotation_mat


def face_alignment(rect, gray, image):
	x = rect.left()
	y = rect.top()
	w = rect.right()
	h = rect.bottom()

	shape = predictor(gray, rect)
	shape = shape_to_normal(shape)
	nose, left_eye, right_eye = get_eyes_nose_dlib(shape)
	center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
	center_pred = (int((x + w) / 2), int((y + y) / 2))
	length_line1 = distance(center_of_forehead, nose)
	length_line2 = distance(center_pred, nose)
	length_line3 = distance(center_pred, center_of_forehead)
	cos_a = cosine_formula(length_line1, length_line2, length_line3)
	angle = np.arccos(cos_a)
	rotated_point = rotate_point(nose, center_of_forehead, angle)
	rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
	if is_between(nose, center_of_forehead, center_pred, rotated_point):
		angle = np.degrees(-angle)
	else:
		angle = np.degrees(angle)

	cvimage, _ = rotateImage(image, angle)

	gray = cv2.cvtColor(cvimage, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	rect = rects[0]
	left = rect.left()
	top = rect.top()
	right = rect.right()
	bottom = rect.bottom()
	img = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(img)
	face = im_pil.crop((left, top, right, bottom)) 
	return face, rect, cvimage



def predict(img_path, model_path="./models/resnet18_best_state.pt"):
	img_name = os.path.basename(img_path)
	img_dir_name = os.path.dirname(img_path)

	resnet_model = load_model(model_arch)
	resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
	resnet_model.eval()

	image = cv2.imread(img_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	if len(rects) == 1:
		rect = rects[0]
		img_cropped, new_rect, image = face_alignment(rect, gray, image)
		img_cropped = transform(img_cropped)
		img_cropped.unsqueeze_(0)
		output = resnet_model(img_cropped)
		print("output", output.item())
		res = round(output.item(), 2)
		return res
	else:
		return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Facial beauty predictor.")
	parser.add_argument('-i', dest='image', help='Image to be predicted.')
	parser.add_argument('-m', dest='model', help='The model path of facial beauty predictor.')
	args = parser.parse_args()
	if args.image and args.model:
		predict(args.image, args.model)