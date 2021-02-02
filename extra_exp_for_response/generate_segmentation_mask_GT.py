from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core._multiarray_umath import ndarray
import scipy.optimize as optimization
# import scipy.ndimage.filters as filters
import scipy.ndimage as nd
from skimage import measure, draw, morphology
import numpy as np
# from utils import *
from scipy.linalg import solve
import cv2
from scipy import spatial
import time
import pdb
import sys
from skimage.color import rgb2hsv
import os
from shapely.geometry import LineString, Polygon, shape, mapping
from scipy import stats
from rasterio.features import rasterize
import drlse
import sknw
from scipy.signal import savgol_filter
import json
from shapely.geometry import Point,LinearRing
# from shapely.geometry import LineString
# parameters

img_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\images'
img_list=os.listdir(img_dir)
gt_mask_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\GT_mask'
GT_dir= r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response'
for img_name in img_list:
	print(img_name)
	prename = img_name.split('.')[0]
	img = np.array(Image.open(os.path.join(img_dir,img_name)).convert('L'))
	[H, W] = img.shape
	img_color=np.array(Image.open(os.path.join(img_dir,img_name)))



	with open(os.path.join(GT_dir, 'GT.json'), "r") as load_f:
		reference_dictload_dict = json.load(load_f)

	reference_geos = []
	for value in reference_dictload_dict.values():
		if value['filename'] == img_name:
			aline=[]
			for _shape in value['regions']:
				
				_polygon = Polygon(zip(_shape['shape_attributes']['all_points_x'],_shape['shape_attributes']['all_points_y']))
				geo = shape(_polygon)
				# geo = geo.buffer(1)
				reference_geos.append((geo, 255))
	reference_img = rasterize(reference_geos, out_shape=(H, W))

	cv2.imwrite(os.path.join(gt_mask_dir, prename + '.png'), reference_img)