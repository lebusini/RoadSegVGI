import json
import os

# from utils import *
import cv2
# import scipy.ndimage.filters as filters
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import Polygon, shape

# from shapely.geometry import LineString
# parameters
flag = 1  # flag为1，则动态展示变化过程
timestep = 2  # time step
mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
max_iter = 300
ev = 100  # 每迭代ev次显示一次变化
# iter_outer = 1
lmda = 5  # coefficient of the weighted length term L(phi)
alfa = -3  # coefficient of the weighted area term A(phi)
epsilon = 1.5  # parameter that specifies the width of the DiracDelta function

c0 = 2
# filename = '216748_216749-100180_100181-18.jpg'
# labelname = '216748_216749-100180_100181-18.npy'



img_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\GT\img'

img_list=os.listdir(img_dir)
road_segmentation_referenc_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\GT\GT'

GT_dir= r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\GT\reference_for_display.json'
for img_name in img_list:
	print(img_name)
	prename = img_name.split('.')[0]
	img = np.array(Image.open(os.path.join(img_dir,img_name)).convert('L'))
	[H, W] = img.shape
	img_color=np.array(Image.open(os.path.join(img_dir,img_name)))



	with open(GT_dir, "r") as load_f:
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

	cv2.imwrite(os.path.join(road_segmentation_referenc_dir, prename + '.png'), reference_img)