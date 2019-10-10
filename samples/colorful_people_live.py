import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from django.conf import settings
settings.configure()

from skimage.color import rgb2gray
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import keyboard
from imagekit.processors import Adjust
from imagekit import ImageSpec
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Configurations
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



# Create window
window = tkinter.Tk()
window.title("Colorful person in a gray world")

# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
# Take picture and release camera
s, cv_img = cam.read()
#cam.release()
height, width, no_channels = cv_img.shape

# Create a canvas that can fit the above image
canvas = tkinter.Canvas(window, width = width, height = height)
canvas.pack()
print("Press x to exit")


while not keyboard.is_pressed('x'):
	#start_time = time.time()
	#print('Start of loop')

	# Take picture 
	s, cv_img = cam.read()
	cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
	
	if s:    # frame captured without any errors
		# Run detection
		results = model.detect([cv_img], verbose=0)
		r = results[0]
		ids = r['class_ids']
		masks = r['masks']
		pers_id = class_names.index("person")
		#print(time.time() - start_time)
		# Check if person exists in image
		if((ids == pers_id).any()):
		    # Get index and mask for person
			pers_index =((ids-pers_id).argmin())
			pers_mask = masks[:,:,pers_index]

			# Make grayscale image
			gray = rgb2gray(cv_img)

			# Create output image
			im_out = cv_img

			# Recolor output image
			for i in range(im_out.shape[0]):
			    for j in range (im_out.shape[1]):
			        if(not pers_mask[i,j]):
			            px_val = gray[i,j]*255
			            im_out[i,j,:] = [px_val, px_val, px_val]
		else:
			im_out = rgb2gray(cv_img)
		#print(time.time() - start_time)
		
		# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
		photo = PIL.ImageTk.PhotoImage(master=canvas,image = PIL.Image.fromarray(im_out))
		# Add a PhotoImage to the Canvas
		canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

	window.update_idletasks()
	window.update()
	#print(time.time() - start_time)

window.quit()
cam.release()	