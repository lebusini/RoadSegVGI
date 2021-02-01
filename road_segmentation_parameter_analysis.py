from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core._multiarray_umath import ndarray
import scipy.optimize as optimization
import sys
sys.path.append('../')
sys.path.append('../..')
from utils.drlse_tools import *
# import scipy.ndimage.filters as filters
import scipy.ndimage as nd
from skimage import measure, draw, morphology,color
import numpy as np
from utils import *
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
from utils.compute_geodesic import get_geodesic_path
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
max_iter = 600
ev = 100  # 每迭代ev次显示一次变化
# iter_outer = 1
lmda = 5  # coefficient of the weighted length term L(phi)
alfa = -3  # coefficient of the weighted area term A(phi)
epsilon = 1.5  # parameter that specifies the width of the DiracDelta function

c0 = 2

img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\parameter_analysis'
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
road_segmentaion_reference_img_dir=r'C:\python_pycharm_label_test\experiment_data\road_segmentation_reference_results_512'
parameter_analysis_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\parameter_analysis'
    
for img_name in img_list:
    prename = img_name.split('.')[0]
    img=Image.open(os.path.join(img_dir, prename+'.jpg'))
    img_gray = np.array(img.convert('L'))
    img_color = np.array(img)
    label_json = np.load(os.path.join(json_dir, prename + '.npy'), allow_pickle=True)
    mask = np.zeros(img_gray.shape)
    filter = np.zeros(img_gray.shape)
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
                linebuffer = line.buffer(4)
                linebuffer1 = line.buffer(25)
                xs, ys = linebuffer.exterior.coords.xy
                rr, cc = draw.polygon(ys, xs, (W, H))
                mask[rr, cc] = 1

                xs, ys = linebuffer1.exterior.coords.xy
                rr, cc = draw.polygon(ys, xs, (W, H))
                filter[rr,cc]=1

        elif _geo[0]['type'] == 'MultiLineString':
            for coor in _geo[0]['coordinates']:

                roadpoints_coordinates = coor

                line = LineString(roadpoints_coordinates)
                if line.length > 50:
                    linebuffer = line.buffer(4)
                    linebuffer1 = line.buffer(25)
                    xs, ys = linebuffer.exterior.coords.xy
                    rr, cc = draw.polygon(ys, xs, (W, H))
                    mask[rr, cc] = 1

                    xs, ys = linebuffer1.exterior.coords.xy
                    rr, cc = draw.polygon(ys, xs, (W, H))
                    filter[rr, cc] = 1

    # ours method wrap
    def ours(H,W,mu,lmda,alfa,n_segments,save_dir,reference_dir,postfix,kernel_size ,sigma):
        '''
        =======================================================================
            drlse
        =======================================================================
        '''

        # 设置道路中心线缓冲区作为初始区域

        phi = np.where(mask==0,2,-2)

        ## save the initial status image


        # start level set evolution

        phi = drlse.drlse(img_gray.astype(np.float32), phi.astype(np.float32), W, H, mu, timestep, lmda, alfa, epsilon, 1,
                          max_iter,kernel_size,sigma)



        kernel = np.ones((10, 10), np.uint8)
        phi_mat_binary = np.where(phi < 0, 255, 0).astype(np.uint8)

        closing = cv2.morphologyEx(phi_mat_binary, cv2.MORPH_CLOSE, kernel)

        filtered_closing=closing*filter




        '''
        =======================================================================
            SLIC
        =======================================================================
        '''
        slic_img = slic(img_color, n_segments, compactness=15, sigma=0.5)
        # plt.imshow(mark_boundaries(img, slic_img))
        # plt.savefig(os.path.join(img_results_dir, prename + 'slic_result.png'))


        merge_location = np.where(filtered_closing != 0)


        merge_label=slic_img[merge_location]

        merge_class_list=np.unique(merge_label)




        '''------- directly compute the union part----------'''
        ravel_slic_img=np.ravel(slic_img)
        final_merge_img=ravel_slic_img.copy().astype(np.uint8)
        for _idx,_val in enumerate (ravel_slic_img):
            if _val in merge_class_list:
                final_merge_img[_idx]=255
            else:
                final_merge_img[_idx]=0
        final_merge_img=final_merge_img.reshape(slic_img.shape)
        closing_final = cv2.morphologyEx(final_merge_img, cv2.MORPH_CLOSE, kernel)
        closing_final = cv2.medianBlur(closing_final, 9)

        cv2.imwrite(os.path.join(save_dir, postfix+ '.png'), closing_final)

        print(prename)


        road_segmentation_ours=closing_final
        road_reference_img=np.array(Image.open(os.path.join(reference_dir, prename + '.png')))


        # Completeness_threeFeatures, Correctness_threeFeatures, Quality_threeFeatures = valuate(road_segmentation_threeFeatures, road_reference_img)
        Precision,Recall,F1,IoU = valuate(road_segmentation_ours, road_reference_img)
        return Precision,Recall,F1,IoU

    # valuate_method
    def valuate(extract_img,reference_img):

        intersection_len=len(np.where(extract_img*reference_img!=0)[0])
        extract_len=len(np.where(extract_img!=0)[0])
        reference_len = len(np.where(reference_img != 0)[0])

        TP=intersection_len
        FN = reference_len - intersection_len
        FP = extract_len - intersection_len
        Recall=0
        if TP + FN!=0:
            Recall = TP/ (TP + FN)
        Precision=0
        if TP + FP!=0:
            Precision = TP / (TP + FP)
        F1=0
        if Precision+Recall!=0:
            F1=2*Precision*Recall/(Precision+Recall)
        Iou=0
        if TP + FP + FN!=0:
            IoU = TP / (TP + FP + FN)
            
        return Precision,Recall,F1,IoU


    # test the experiments for the parameters
    n_segments=1000
    # mu
    mu_test_array=np.zeros((16,5))
    for idx,val in enumerate(np.linspace(start = 0, stop = 0.14, num = 15)):
        mu=val
        _Precision,_Recall,_F1,_IoU=ours(H,W,mu,lmda,alfa,n_segments,parameter_analysis_dir,road_segmentaion_reference_img_dir,prename+'_mu='+str(mu),9,0.8)
        mu_test_array[idx]=np.array([val,_Precision,_Recall,_F1,_IoU])

    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    max_iter = 600
    ev = 100  # 每迭代ev次显示一次变化
    # iter_outer = 1
    lmda = 5  # coefficient of the weighted length term L(phi)
    alfa = -3  # coefficient of the weighted area term A(phi)

    # lmda
    lmda_test_array=np.zeros((11,5))
    for idx,val in enumerate(np.linspace(start = 0, stop = 10, num = 11)):
        lmda=val
        _Precision,_Recall,_F1,_IoU=ours(H,W,mu,lmda,alfa,n_segments,parameter_analysis_dir,road_segmentaion_reference_img_dir,prename+'_lmda='+str(lmda),9,0.8)
        lmda_test_array[idx]=np.array([val,_Precision,_Recall,_F1,_IoU])

    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    max_iter = 600
    ev = 100  # 每迭代ev次显示一次变化
    # iter_outer = 1
    lmda = 5  # coefficient of the weighted length term L(phi)
    alfa = -3  # coefficient of the weighted area term A(phi)

    # alfa
    alfa_test_array=np.zeros((11,5))
    for idx,val in enumerate(np.linspace(start = 0, stop = -10, num = 11)):
        alfa=val
        _Precision,_Recall,_F1,_IoU=ours(H,W,mu,lmda,alfa,n_segments,parameter_analysis_dir,road_segmentaion_reference_img_dir,prename+'_alfa='+str(alfa),9,0.8)
        alfa_test_array[idx]=np.array([val,_Precision,_Recall,_F1,_IoU])

    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    max_iter = 600
    ev = 100  # 每迭代ev次显示一次变化
    # iter_outer = 1
    lmda = 5  # coefficient of the weighted length term L(phi)
    alfa = -3  # coefficient of the weighted area term A(phi)

    # K
    K_test_array=np.zeros((25,5))
    for idx,val in enumerate(np.linspace(start =200, stop = 5000, num = 25)):
        n_segments=val
        _Precision,_Recall,_F1,_IoU=ours(H,W,mu,lmda,alfa,n_segments,parameter_analysis_dir,road_segmentaion_reference_img_dir,prename+'_K='+str(n_segments),9,0.8)
        K_test_array[idx]=np.array([val,_Precision,_Recall,_F1,_IoU])


    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    max_iter = 600
    ev = 100  # 每迭代ev次显示一次变化
    # iter_outer = 1
    lmda = 5  # coefficient of the weighted length term L(phi)
    alfa = -3  # coefficient of the weighted area term A(phi)


    # kernel_size
    kenel_test_array=np.zeros((19,5))
    for idx,val in enumerate(np.linspace(start = 3, stop = 39, num = 19)):
        kenel_size=val
        _Precision,_Recall,_F1,_IoU=ours(H,W,mu,lmda,alfa,n_segments,parameter_analysis_dir,road_segmentaion_reference_img_dir,prename+'_kernel_size='+str(kenel_size),kenel_size,0.8)
        kenel_test_array[idx]=np.array([val,_Precision,_Recall,_F1,_IoU])

    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    max_iter = 600
    ev = 100  # 每迭代ev次显示一次变化
    # iter_outer = 1
    lmda = 5  # coefficient of the weighted length term L(phi)
    alfa = -3  # coefficient of the weighted area term A(phi)
    # sigma
    sigma_test_array=np.zeros((26,5))
    for idx,val in enumerate(np.linspace(start = 0, stop = 5, num = 26)):
        sigma=val
        _Precision,_Recall,_F1,_IoU=ours(H,W,mu,lmda,alfa,n_segments,parameter_analysis_dir,road_segmentaion_reference_img_dir,prename+'_sigma='+str(sigma),9,sigma)
        sigma_test_array[idx]=np.array([val,_Precision,_Recall,_F1,_IoU])

    # mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    # max_iter = 600
    # ev = 100  # 每迭代ev次显示一次变化
    # # iter_outer = 1
    # lmda = 5  # coefficient of the weighted length term L(phi)
    # alfa = -3  # coefficient of the weighted area term A(phi)
    # # iters
    # iters_test_array=np.zeros((12,4))
    # for idx,val in enumerate(np.linspace(start = 100, stop = 1200, num = 6)):
    #     max_iter=val
    #     _Completeness_ours, _Correctness_ours, _Quality_ours=ours(H,W,mu,lmda,alfa,n_segments,parameter_analysis_dir,road_segmentaion_reference_img_dir,prename+'_iters='+str(max_iter),9,0.8)
    #     iters_test_array[idx]=np.array([val,_Completeness_ours, _Correctness_ours, _Quality_ours])



    np.save(prename+'_mu.npy',mu_test_array)
    np.save(prename+'_lmda.npy',lmda_test_array)
    np.save(prename+'_alfa.npy',alfa_test_array)
    np.save(prename+'_n_segments.npy',K_test_array)
    np.save(prename+'_sigma.npy',sigma_test_array)
    np.save(prename+'_kenel.npy',kenel_test_array)
    # np.save(prename + '_iters.npy', iters_test_array)