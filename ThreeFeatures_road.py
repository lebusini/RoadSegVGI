from shapely.geometry import LineString
from PIL import ImageFilter, Image
# import scipy.ndimage.filters as filters
import scipy.ndimage as nd
from skimage import measure, draw
import numpy as np
from skimage.color import rgb2hsv
import time
from skimage.filters import threshold_otsu
from scipy import stats, linalg
from compute_geodesic import get_geodesic_path, normalize_im
# from skimage import data
import cv2 as cv
from scipy import signal
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from drlse_tools import *
from sklearn.cluster import MeanShift, estimate_bandwidth
# 初始种子点[153,504] [510,245]
from shapely.geometry import Point, Polygon
from GuidedFilter.core.filter import GuidedFilter
import datetime
# from shapely.geometry import LineString

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
img_results_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\three_features'

radius = 2
eps = 0.005
alafa=0.9
beta=0.7
lamda=0.5
c0=2

if __name__ == '__main__':

	with open(os.path.join(exp_resluts_dir, 'experiment_result.txt'), 'a') as f:
		f.write("\n\n")
		f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')  # print the current time
		f.write('ThreeFeatures_road \n')  # print the method name
		f.write("\n\n")
		f.write("name        time   \n")

		for img_name in img_list:
			start = time.time()
			prename = img_name.split('.')[0]
			img = np.array(Image.open(os.path.join(img_dir, img_name)).convert('L'))
			img_array = np.array(Image.open(os.path.join(img_dir, img_name)))
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



			GF = GuidedFilter(img_array, radius, eps)
			guided_filtered_img = GF.filter(img_array)

			[H, W] = img_array.shape[0:2]




			# 计算马氏距离
			def mahalanobis(Y, X):
				"""D2 = MAHAL(Y,X) returns the Mahalanobis distance (in squared units) of
				each observation (point) in Y from the sample data in X, i.e.,
				D2(I) = (Y(I,:)-MU) * SIGMA^(-1) * (Y(I,:)-MU)'
				"""
				[rx, cx] = X.shape
				[ry, cy] = Y.shape
				m = np.mean(X, 0)
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
			seed_area = guided_filtered_img[rr, cc]
			R = guided_filtered_img[:, :, 0]
			G=guided_filtered_img[:,:,1]
			B=guided_filtered_img[:,:,2]
			subset_R = seed_area[:, 0]
			subset_G=seed_area[:,1]
			subset_B=seed_area[:,2]
			img_data=np.concatenate((R.reshape(-1,1),G.reshape(-1,1),B.reshape(-1,1)),axis=1)

			sub_data=np.concatenate((subset_R.reshape(-1,1),subset_G.reshape(-1,1),subset_B.reshape(-1,1)),axis=1)
			# sub_data=np.hstack((tmp,subset_B.reshape(-1,1)))

			mahal = mahalanobis(img_data, sub_data)
			# normal_mahal = (255 * MaxMinNormalization(mahal)).astype(np.uint8)
			mahal_mat = mahal.reshape(H,W)
			normal_mahal_mat = MaxMinNormalization(mahal_mat)

			# 图像二值化，或得道路域分割结果
			# image = data.camera()
			# thresh = threshold_otsu(mahal_mat)
			road_binary = normal_mahal_mat < 0.3



			#计算光谱图像
			# sigma = 3.0
			# k=3
			def S(i,j,k,sigma):
				Sgauss=1/(2*np.pi*sigma**2)*np.exp(-1*((-i-k-1)**2-(j-k-1)**2)/(2*sigma**2))
				return Sgauss
			gauss_filter=np.zeros((3,3))
			gauss_filter[0,0]=S(-1,-1,3,3)
			gauss_filter[0,1]=S(-1,0,3,3)
			gauss_filter[0,2]=S(-1,1,3,3)
			gauss_filter[1,0]=S(0,-1,3,3)
			gauss_filter[1,1]=S(0,0,3,3)
			gauss_filter[1,2]=S(0,1,3,3)
			gauss_filter[2,0]=S(1,-1,3,3)
			gauss_filter[2,1]=S(1,0,3,3)
			gauss_filter[2,2]=S(1,1,3,3)

			gauss_filter=gauss_filter/np.sum(gauss_filter)
			# spectral_img = nd.convolve(guided_filtered_img[:, :, 0].astype(np.float), gauss_filter, mode='constant')
			spectral_img = nd.convolve(road_binary.astype(np.float), gauss_filter, mode='constant')


			# distance transform
			img_distance_transfrom_edt = nd.distance_transform_edt(road_binary)
			img_distance_transfrom_edt = MaxMinNormalization(img_distance_transfrom_edt)


			# 滑动窗口求能量边缘
			def SA(v, w):
				cos=1
				if np.linalg.norm(v) * np.linalg.norm(w)!=0:
					cos=np.round_(np.sum(v * w) / (np.linalg.norm(v) * np.linalg.norm(w)),5)
				value=np.arccos(cos)
				return value

			stepSize = 1
			(w_width, w_height) = (3, 3)
			energy_edge_img = np.zeros(guided_filtered_img.shape[0:2])
			expanded_image_mat = np.zeros(
				(guided_filtered_img.shape[0] + 2, guided_filtered_img.shape[1] + 2, guided_filtered_img.shape[2]))
			expanded_image_mat[1:-1, 1:-1, :] = img_array
			expanded_image_mat[:, [0, -1], :] = expanded_image_mat[:, [1, -2], :]
			expanded_image_mat[[0, -1], :, :] = expanded_image_mat[[1, -2], :, :]
			for x in range(1, guided_filtered_img.shape[1]+1, stepSize):
				for y in range(1, guided_filtered_img.shape[0]+1, stepSize):
					center_value=expanded_image_mat[y, x, :].reshape(3,-1)
					value1 = SA(expanded_image_mat[y-1, x-1, :].reshape(3,-1),center_value)
					value2 = SA(expanded_image_mat[y - 1, x, :].reshape(3,-1),center_value)
					value3 = SA(expanded_image_mat[y - 1, x + 1, :].reshape(3,-1),center_value)
					value4 = SA(expanded_image_mat[y, x - 1, :].reshape(3,-1),center_value)
					value5 = SA(expanded_image_mat[y, x + 1, :].reshape(3,-1),center_value)
					value6 = SA(expanded_image_mat[y + 1, x - 1, :].reshape(3,-1),center_value)
					value7 = SA(expanded_image_mat[y + 1, x, :].reshape(3,-1),center_value)
					value8 = SA(expanded_image_mat[y + 1, x + 1, :].reshape(3,-1),center_value)
					if value1+2*value2+value3+2*value4+2*value5+value6+2*value7+value8!=0:
						energy_edge_img[y-1, x-1] = 1/12*(value1+2*value2+value3+2*value4+2*value5+value6+2*value7+value8)
					else:
						energy_edge_img[y-1, x-1] = 1
			energy_edge_img=MaxMinNormalization(energy_edge_img)


			# energy_edge_img=np.load('energy_edge_img.npy')
			# 核密度估计(这里执行时间很长,采用线程的话需要五分钟)
			roadarea = np.where(road_binary == True)
			roadpositions = np.vstack([roadarea[0], roadarea[1]])  # 道路区域
			kernel = stats.gaussian_kde(roadpositions.copy())
			# Create definition.
			def calc_kernel(samp):
				return kernel(samp)


			# Calculate

			tik = time.time()
			cores = 4
			torun = np.array_split(roadpositions.copy(), cores, axis=1)
			pool = Pool(processes=cores)
			results = pool.map(kernel, torun)
			kdearray = np.concatenate(results)
			print('multiprocessing filter/sample: ', time.time() - tik)
			kde_img = np.ones(img_array.shape[0:2])
			kde_img[roadpositions[0], roadpositions[1]] = kdearray


			def convolution(img):

				kernel = (3, 3)  # 腐蚀膨胀的核
				# 开运算，先腐蚀后膨胀
				img2 = cv2.erode(img, kernel)  # 腐蚀操作
				img3 = cv2.dilate(img2, kernel)  # 膨胀操作
				# 闭运算，先膨胀后腐蚀
				img4 = cv2.dilate(img3, kernel)  # 膨胀操作
				img5 = cv2.erode(img4, kernel)  # 腐蚀操作
				a = img5.shape[0]  # 获取图像的行数
				b = img5.shape[1]  # 获取图像的列数
				img6=[]
				gray5=[]
				gray6=[]
				if len(img5.shape)==3:
					c = img5.shape[2]  # 获取图像的通道数
					img6 = cv2.copyMakeBorder(img5, 1, 1, 1, 1, cv2.BORDER_DEFAULT, (0, 0, 0))
					gray5 = cv2.cvtColor(img5, cv2.COLOR_RGB2GRAY)
					gray6 = cv2.cvtColor(img6, cv2.COLOR_RGB2GRAY)
				else:
					img6 = cv2.copyMakeBorder(img5, 1, 1, 1, 1, cv2.BORDER_DEFAULT, 0)
					gray5 = img5
					gray6 = img6
				a2 = gray6.shape[0]
				b2 = gray6.shape[1]
				dst = gray5

				# 进行曲率提取操作
				for i in range(1, a2 - 1):
					for j in range(1, b2 - 1):
						dst[i - 1, j - 1] = (-gray6[i - 1, j - 1] + 5 * gray6[i - 1, j] - gray6[i - 1, j + 1] + 5 * gray6[
							i, j - 1] - 16 * gray6[i, j] + 5 * gray6[i, j + 1] - gray6[i + 1, j - 1] + 5 * gray6[i + 1, j] -
											 gray6[i + 1, j + 1]) / 16
				return dst


			# # 检测二值图像的轮廓
			# contours = measure.find_contours(road_binary, 0)
			# edge_positions=np.concatenate(contours)
			# kde_distance_img=np.zeros(road_binary.shape)
			# kernel = stats.gaussian_kde(edge_positions.copy().T)


			# # Calculate

			# tik = time.time()
			# cores = 4
			# torun = np.array_split(edge_positions.copy().T, cores, axis=1)
			# pool = Pool(processes=cores)
			# results = pool.map(kernel, torun)
			# kdearray = np.concatenate(results)
			# print('multiprocessing filter/sample: ', time.time() - tik)
			# kde_distance_img[edge_positions[:,0].astype(np.int), edge_positions[:,1].astype(np.int)] =1-kdearray


			curvature_img = convolution(energy_edge_img)
			curvature_img =np.abs(curvature_img)
			curve_positons_zeros = np.where(curvature_img == 0)
			curvature_img[curve_positons_zeros]=0.1

			edge_item=np.ones(road_binary.shape)
			edge_item[curve_positons_zeros]=0
			edge_item=edge_item*lamda*(kde_img-1)*energy_edge_img/curvature_img
			edge_item=-edge_item/np.sum(edge_item)

			statics_img=alafa*spectral_img+beta*img_distance_transfrom_edt+edge_item

			statics_img=((statics_img/(np.max(statics_img)-np.min(statics_img)))*255).astype(np.uint8)
			ret1, th1 = cv2.threshold(statics_img, 0, 255, cv2.THRESH_OTSU)
			median_val=np.median(statics_img)

			seg_img_median=(statics_img>median_val)*255
			seg_img_otsu = (statics_img > ret1) * 255
			end = time.time()
			dur_time = end - start
			print(prename)


			f.write("%s        %d   \n" % (prename,dur_time))

			cv2.imwrite(os.path.join(img_results_dir, prename + '_median.png'), seg_img_median)
			cv2.imwrite(os.path.join(img_results_dir, prename + '_otsu.png'), seg_img_otsu)
			cv2.imwrite(os.path.join(img_results_dir, prename + '_statics.png'), statics_img)

