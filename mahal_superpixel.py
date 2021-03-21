from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core._multiarray_umath import ndarray
import scipy.optimize as optimization
from utils_tools.drlse_tools import *
# import scipy.ndimage.filters as filters
import scipy.ndimage as nd
from skimage import measure, draw, morphology,color
import numpy as np
from utils_tools import *
from scipy.linalg import solve
from scipy.signal import argrelextrema
import cv2 as cv
from scipy import spatial, stats
import time
import pdb
import sys
from skimage.color import rgb2hsv
from skimage.segmentation import slic,mark_boundaries,find_boundaries
from shapely.geometry import LineString, Polygon, shape, mapping
from utils_tools.compute_geodesic import get_geodesic_path
from rasterio.features import rasterize
import drlse
import sknw
from scipy.signal import savgol_filter
import json
import datetime
from shapely.geometry import Point, LinearRing

# from shapely.geometry import LineString
# parameters
flag = 1  # flag为1，则动态展示变化过程
timestep = 2  # time step
mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
max_iter = 500
ev = 100  # 每迭代ev次显示一次变化
# iter_outer = 1
lmda = 5  # coefficient of the weighted length term L(phi)
alfa = -3  # coefficient of the weighted area term A(phi)
epsilon = 1.5  # parameter that specifies the width of the DiracDelta function

c0 = 2

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
img_results_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\ours'

with open(os.path.join(exp_resluts_dir, 'experiment_result.txt'), 'a') as f:
    f.write("\n\n")
    f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')  # print the current time
    f.write('ours_method \n')  # print the method name
    f.write("\n\n")
    f.write("name        time   \n")
    for img_name in img_list:
        start = time.time()
        prename = img_name.split('.')[0]
        img=Image.open(os.path.join(img_dir, img_name))
        img_gray = np.array(img.convert('L'))
        img_color = np.array(img)
        label_json = np.load(os.path.join(json_dir, prename + '.npy'), allow_pickle=True)
        mask = np.zeros(img_gray.shape)
        savename = prename + '.png'
        [H, W] = img_gray.shape



        '''
        ====================================================================================================
            initiate the initial area in the image
        =====================================================================================================
        '''

        for _geo in label_json:
            if _geo[0]['type'] == 'LineString':
                # 设置道路中心线缓冲区作为初始区域

                roadpoints_coordinates = _geo[0]['coordinates']

                line = LineString(roadpoints_coordinates)
                if line.length > 50:  # remove the small ones
                    linebuffer = line.buffer(25)
                    xs, ys = linebuffer.exterior.coords.xy
                    rr, cc = draw.polygon(ys, xs, (W, H))
                    mask[rr, cc] = 1

            elif _geo[0]['type'] == 'MultiLineString':
                for coor in _geo[0]['coordinates']:

                    roadpoints_coordinates = coor

                    line = LineString(roadpoints_coordinates)
                    if line.length > 50:
                        linebuffer = line.buffer(25)
                        xs, ys = linebuffer.exterior.coords.xy
                        rr, cc = draw.polygon(ys, xs, (W, H))
                        mask[rr, cc] = 1


        '''
        =======================================================================
            mahalanobis
        =======================================================================
        '''

        mahal = mahalanobis(img_color.reshape(-1, 3), img_color[np.where(mask==1)].reshape(-1, 3))
        normal_mahal = (255 * MaxMinNormalization(mahal)).astype(np.uint8)

        img_mat = normal_mahal.reshape(H, W)
        img_seg = (img_mat < 50) * 255
        mahal_mask_seg=(mask*img_seg).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)

        '''
        ====================================================================================================
            morphological method to process the road area
        =====================================================================================================
        '''
        opening = cv2.morphologyEx(mahal_mask_seg, cv2.MORPH_OPEN, kernel)


        '''
        =======================================================================
            remove too small region
        =======================================================================
        '''
        morphological_image=opening.copy()
        label_region_image = measure.label(morphological_image, connectivity=2)
        road_binary_regions = measure.regionprops(label_region_image)
        # remove very small road region
        for _region in road_binary_regions:
            if _region.area <300:
                morphological_image[_region.coords[:, 0], _region.coords[:, 1]] = 0
        kernel1 = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(mahal_mask_seg, cv2.MORPH_CLOSE, kernel1)
        cv2.imwrite(os.path.join(img_results_dir, prename + 'p.png'), morphological_image)

        '''
        =======================================================================
            fill the holes
        =======================================================================
        '''
        revese_image=np.where(morphological_image==0,255,0)
        label_region_image_rev = measure.label(revese_image, connectivity=2)
        road_binary_regions_rev = measure.regionprops(label_region_image_rev)
        # remove very small road region
        for _region in road_binary_regions_rev:
            if _region.area < 300:
                revese_image[_region.coords[:, 0], _region.coords[:, 1]] = 0

        final_morphological_image=np.where(revese_image==0,255,0)
        cv2.imwrite(os.path.join(img_results_dir, prename + '.png'), final_morphological_image)

        # cv2.imwrite(os.path.join(img_results_dir, prename + '.png'), img_seg)

        # dilation_img = morphology.binary_dilation(phi_mat_binary, morphology.disk(10))
        # dilation_img = cv2.medianBlur((dilation_img * 255).astype(np.uint8), 31)
        # kernel = np.ones((5, 5), np.uint8)
        # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # kernel = np.ones((5, 5), np.uint8)
        # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite(os.path.join(phi_binary_result_dir, prename + '.png'), dilation_img)


        # '''
        # =======================================================================
        #     SLIC
        # =======================================================================
        # '''
        # img_lab = color.rgb2lab(img)
        # [H, W, Ch] = img_color.shape
        # data = np.reshape(img_color, (H * W, Ch))
        #
        # slic_img = slic(img_color, n_segments=1000, compactness=10, sigma=1)
        #
        # # plt.imshow(mark_boundaries(img, slic_img))
        # merge_location=np.where(morphological_image!=0)
        #
        # merge_label=slic_img[merge_location]
        #
        # merge_class_list=np.unique(merge_label)
        #
        # slic_class_list=np.unique(slic_img)
        #
        # '''
        # ====================================================================================
        #     computhe the union part of slic results and the segmentation meraging result,
        #     when the union is over half of the number of the slic superpixels
        # ==================================================================================
        # '''

        # final_merge_img=slic_img.copy()
        # for _cls in slic_class_list:
        #     merge_location=np.where(merge_label==_cls)
        #     slic_location=np.where(slic_img==_cls)
        #     _occupy=len(merge_location[0])/len(slic_location[0])
        #     if (_cls in merge_class_list) and (_occupy>0.5): # the class label in the union part and the occupation is over 0.5
        #         final_merge_img[slic_location]=255
        #     else:
        #         final_merge_img[slic_location] = 0
        # final_merge_img=final_merge_img.astype(np.uint8)

        # '''------- directly compute the union part----------'''
        # ravel_slic_img=np.ravel(slic_img)
        # final_merge_img=ravel_slic_img.copy().astype(np.uint8)
        # for _idx,_val in enumerate (ravel_slic_img):
        #     if _val in merge_class_list:
        #         final_merge_img[_idx]=255
        #     else:
        #         final_merge_img[_idx]=0
        # final_merge_img=final_merge_img.reshape(slic_img.shape)


        # cv2.imwrite(os.path.join(img_results_dir, prename + '.png'), final_merge_img)


        end = time.time()
        dur_time = end - start
        f.write("%s        %d   \n" % (prename, dur_time))

        print(prename)


 



