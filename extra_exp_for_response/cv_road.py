from shapely.geometry import LineString
from PIL import ImageFilter, Image, ImageDraw
# import scipy.ndimage.filters as filters
import scipy.ndimage as nd
from skimage import measure, draw
import numpy as np
from skimage.color import rgb2hsv
import datetime
from scipy import stats
# from compute_geodesic import get_geodesic_path,normalize_im
from chanvese import chanvese
import cv2 as cv
from scipy import signal
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from drlse_tools import *
import time
# 初始种子点[153,504] [510,245]

# from shapely.geometry import Point
# from shapely.geometry import LineString
# parameters

# mapbox: sigma=3.0 alpha=2 kernel_size=30 max_its=600
# yahoo,google: sigma=3.0 alpha=1 kernel_size=30 max_its=500
sigma = 3.0  # scale parameter in Gaussian kernel
alpha=1
kernel_size=30
c0 = 2
max_its=500


img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\images'

json_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\vectors'

exp_resluts_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response'

img_results_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\predictions\cv_road'

img_list = os.listdir(img_dir)
json_list = os.listdir(img_dir)



with open(os.path.join(exp_resluts_dir, 'experiment_result.txt'), 'a') as f:
	f.write("\n\n")
	f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')  # print the current time
	f.write('cv_road \n')  # print the method name
	f.write("\n\n")
	f.write("name        time   \n")

	for img_name in img_list:
		start = time.time()
		prename = img_name.split('.')[0]
		img = np.array(Image.open(os.path.join(img_dir, img_name)).convert('L'))
		img_color = np.array(Image.open(os.path.join(img_dir, img_name)))
		label_json = np.load(os.path.join(json_dir, prename + '.npy'), allow_pickle=True)

		savename = prename + '.png'
		[H, W] = img.shape

		mask = np.zeros((W, H))
		for _geo in label_json:
			if _geo[0]['type'] == 'LineString':
				# 设置道路中心线缓冲区作为初始区域

				roadpoints_coordinates = _geo[0]['coordinates']

				line = LineString(roadpoints_coordinates)
				if line.length > 50:  # remove the small ones
					linebuffer = line.buffer(4)
					xs, ys = linebuffer.exterior.coords.xy
					rr, cc = draw.polygon(ys, xs, (W, H))
					mask[rr, cc] = 1

			elif _geo[0]['type'] == 'MultiLineString':
				for coor in _geo[0]['coordinates']:

					roadpoints_coordinates = coor

					line = LineString(roadpoints_coordinates)
					if line.length > 50:
						linebuffer = line.buffer(4)
						xs, ys = linebuffer.exterior.coords.xy
						rr, cc = draw.polygon(ys, xs, (W, H))
						mask[rr, cc] = 1



		# HSV图像
		hsv_img = rgb2hsv(img_color)
		s_img = hsv_img[:, :, 1]

		##图像预处理：图像平滑
		gray_img =s_img
		G = matlab_style_gauss2D((kernel_size, kernel_size), sigma)

		# img_smooth = img.filter(ImageFilter.GaussianBlur(radius=30))
		img_smooth = nd.convolve(gray_img.astype(np.float32), G, mode='constant')

		seg, phi, its=chanvese(img_smooth, mask, max_its,alpha,  display=False,  show_img=gray_img)

		phi_mat_binary = np.where(phi < 0, 255, 0).astype(np.uint8)

		end = time.time()
		dur_time = end - start
		print(prename)

		f.write("%s        %d   \n" % (prename, dur_time))

		cv2.imwrite(os.path.join(img_results_dir, prename + '.png'), phi_mat_binary)
	

