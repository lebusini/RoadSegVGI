from shapely.geometry import LineString
from PIL import ImageFilter, Image
# import scipy.ndimage.filters as filters
import scipy.ndimage as nd
from skimage import measure, draw
import numpy as np
from skimage.color import rgb2hsv
import time
import datetime
from scipy import stats, linalg
from utils_tools.compute_geodesic import get_geodesic_path, normalize_im
import cv2 as cv
from scipy import signal
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from utils_tools.drlse_tools import *
# 初始种子点[153,504] [510,245]

from shapely.geometry import Point, Polygon, LineString


img_dir = r'C:\python_pycharm_label_test\experiment_data\img_tile_512'
json_dir = r'C:\python_pycharm_label_test\experiment_data\label_json_512'
phi_binary_result_dir = r'C:\python_pycharm_label_test\experiment_data\phi_binary_results_512'
phi_binary_result_dir_ori = r'C:\python_pycharm_label_test\experiment_data\phi_binary_results_512_origin'
road_network_npy_result_dir = r'C:\python_pycharm_label_test\experiment_data\road_network_npy_result_512'
road_network_img_result_dir = r'C:\python_pycharm_label_test\experiment_data\road_network_img_result_512'
road_network_preview_dir = r'C:\python_pycharm_label_test\experiment_data\road_network_preview_result_512'
# initial_phi_binary_dir=r'C:\python_pycharm_label_test\experiment_data\initial_phi_binary_results'
# buffer_filter_binary_dir=r'C:\python_pycharm_label_test\experiment_data\buffer_filter_binary_512'
exp_resluts_dir = r'C:\python_pycharm_label_test\experiment_data\exp_results_512'
img_list = os.listdir(img_dir)
json_list = os.listdir(img_dir)
phi_reference_512 = r'C:\python_pycharm_label_test\experiment_data\phi_reference_results_512'
phi_skeleton_result_dir = r'C:\python_pycharm_label_test\experiment_data\phi_skeleton_results_512'
phi_thinned_result_dir = r'C:\python_pycharm_label_test\experiment_data\phi_thinned_results_512'
GT_dir = r'C:\python_pycharm_label_test\ground_truth'
label_single_dir=r'C:\python_pycharm_label_test\experiment_data\lebel_single_line_512'
img_results_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\mahal'
c0 = 2

with open(os.path.join(exp_resluts_dir, 'experiment_result.txt'), 'a') as f:
	f.write("\n\n")
	f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')  # print the current time
	f.write('Mahalanobis_road \n')  # print the method name
	f.write("\n\n")
	f.write("name        time   \n")

	for img_name in img_list:
		start = time.time()
		prename = img_name.split('.')[0]
		img = np.array(Image.open(os.path.join(img_dir, img_name)).convert('L'))
		img_color = np.array(Image.open(os.path.join(img_dir, img_name)))
		label_json = np.load(os.path.join(json_dir, prename + '.npy'), allow_pickle=True)
		initialLSF = c0 * np.ones(img.shape)
		savename = prename + '.png'
		[H, W] = img.shape


		for _geo in label_json:
			if _geo[0]['type'] == 'LineString':
				# 设置道路中心线缓冲区作为初始区域

				roadpoints_coordinates = _geo[0]['coordinates']

				line = LineString(roadpoints_coordinates)
				if line.length > 50:  # remove the small ones
					linebuffer = line.buffer(4)
					xs, ys = linebuffer.exterior.coords.xy
					rr, cc = draw.polygon(ys, xs, (W, H))
					initialLSF[rr, cc] = -c0

			elif _geo[0]['type'] == 'MultiLineString':
				for coor in _geo[0]['coordinates']:

					roadpoints_coordinates = coor

					line = LineString(roadpoints_coordinates)
					if line.length > 50:
						linebuffer = line.buffer(4)
						xs, ys = linebuffer.exterior.coords.xy
						rr, cc = draw.polygon(ys, xs, (W, H))
						initialLSF[rr, cc] = -c0

		phi = initialLSF.copy()

		gray_img = img

		# label = np.array(Image.open(labelname))
		# savename = "%s_mu=%.2f_max_iter=%d_lmda=%.2f_alfa=%.2f_epsilon=%.2f_sigma=%.2f_c0=%d.png" % (
		# filename, mu, max_iter, lmda, alfa, epsilon, sigma, c0)
		[H, W] = gray_img.shape


		# 计算马氏距离
		def mahalanobis(Y, X):
			"""D2 = MAHAL(Y,X) returns the Mahalanobis distance (in squared units) of
			each observation (point) in Y from the sample data in X, i.e.,
			D2(I) = (Y(I,:)-MU) * SIGMA^(-1) * (Y(I,:)-MU)'
			"""
			[rx, cx] = X.shape;
			[ry, cy] = Y.shape;
			m = np.mean(X, 0);
			M = m.reshape(1, -1).repeat(ry, axis=0)
			C = X - m.reshape(1, -1).repeat(rx, axis=0)
			Q, R = np.linalg.qr(C)

			ri = np.dot(np.linalg.inv(R.T), (Y - M).T)
			d = np.sum(ri * ri, 0).T * (rx - 1)
			return np.sqrt(d)


		def MaxMinNormalization(x):
			Max = np.max(x)
			Min = np.min(x)
			x = (x - Min) / (Max - Min)
			return x




		# 计算图像的马氏距离矩阵

		mahal = mahalanobis(img_color.reshape(-1, 3), img_color[np.where(phi<0)].reshape(-1, 3))
		normal_mahal = (255 * MaxMinNormalization(mahal)).astype(np.uint8)


		img_mat = normal_mahal.reshape(H,W)
		ret1, th1 = cv2.threshold(img_mat, 0, 255, cv2.THRESH_OTSU)
		img_seg=(img_mat<ret1)*255
		end = time.time()
		dur_time = end - start
		print(prename)

		f.write("%s        %d   \n" % (prename, dur_time))


		cv2.imwrite(os.path.join(img_results_dir, prename + '.png'), img_seg)
	
	
	
	





