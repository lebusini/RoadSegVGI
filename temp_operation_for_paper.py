from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core._multiarray_umath import ndarray
import scipy.optimize as optimization
from drlse_tools import *
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
from compute_geodesic import get_geodesic_path
from rasterio.features import rasterize
import drlse
import sknw
from scipy.signal import savgol_filter
import json
import datetime
from shapely.geometry import Point, LinearRing

img_name='216802_216803-100050_100051-18.jpg'
# from shapely.geometry import LineString
# parameters google,mapbox,yahoo mu=0.02 lmda=6 alfa=-3
flag = 1  # flag为1，则动态展示变化过程
timestep = 2  # time step
mu = 0.02 / timestep  # coefficient of the distance regularization term R(phi)
max_iter = 600
# iter_outer = 1
# mapbox lmda=6
lmda = 6  # coefficient of the weighted length term L(phi)
# mapbox alfa=-3
alfa = -3  # coefficient of the weighted area term A(phi)
epsilon = 1.5  # parameter that specifies the width of the DiracDelta function

c0 = 2

img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\mapbox'
save_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\workflow_GA_images'
json_dir = r'C:\python_pycharm_label_test\experiment_data\label_json_512'
phi_binary_result_dir = r'C:\python_pycharm_label_test\experiment_data\phi_binary_results_512'
phi_binary_result_dir_ori = r'C:\python_pycharm_label_test\experiment_data\phi_binary_results_512_origin'
road_network_npy_result_dir = r'C:\python_pycharm_label_test\experiment_data\road_network_npy_result_512'
road_network_img_result_dir = r'C:\python_pycharm_label_test\experiment_data\road_network_img_result_512'
road_network_preview_dir = r'C:\python_pycharm_label_test\experiment_data\road_network_preview_result_512'
exp_resluts_dir = r'C:\python_pycharm_label_test\experiment_data\exp_results_512'
img_list = os.listdir(img_dir)
phi_reference_512 = r'C:\python_pycharm_label_test\experiment_data\phi_reference_results_512'
phi_skeleton_result_dir = r'C:\python_pycharm_label_test\experiment_data\phi_skeleton_results_512'
phi_thinned_result_dir = r'C:\python_pycharm_label_test\experiment_data\phi_thinned_results_512'
reference_dir = r'C:\python_pycharm_label_test\ground_truth'
label_single_dir=r'C:\python_pycharm_label_test\experiment_data\lebel_single_line_512'
img_results_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\mapbox\ours'




prename = img_name.split('.')[0]
img=Image.open(os.path.join(img_dir, img_name))
img_gray = np.array(img.convert('L'))
img_color = np.array(img)
label_json = np.load(os.path.join(json_dir, prename + '.npy'), allow_pickle=True)
mask = np.zeros(img_gray.shape)
W,H=img_gray.shape



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

'''
=======================================================================
    drlse
=======================================================================
'''

ours_maskarr=(mask*255).astype(np.uint8)
dpi = 300
fig = plt.figure(figsize=(3.5, 3.5), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
axes = fig.add_axes([0, 0, 1, 1])
axes.set_axis_off()
axes.imshow(img_color)

# draw line

for _geo in label_json:
    if _geo[0]['type'] == 'LineString':
        # 设置道路中心线缓冲区作为初始区域

        roadpoints_coordinates = _geo[0]['coordinates']

        line = LineString(roadpoints_coordinates)
        road_x,road_y=line.xy
        plt.plot(road_x,road_y,'#bf360c',linewidth=1)

fig.savefig(os.path.join(save_dir,prename+'_initial1.tif'))


# the initinal over image
ours_maskarr=(mask*255).astype(np.uint8)
dpi = 96
fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
axes = fig.add_axes([0, 0, 1, 1])
axes.set_axis_off()
axes.imshow(img_color)

axes1 = fig.add_axes([0, 0, 1, 1])
axes1.set_axis_off()
mask_rgba2 = np.ones(([H, W, 4]))
mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
mask_rgba2[:, :, 1:3] = 0
alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
mask_rgba2[:, :, -1] = alpha
axes1.imshow(mask_rgba2)
fig.savefig(os.path.join(save_dir,prename+'_initial.jpg'))

# 设置道路中心线缓冲区作为初始区域

phi = np.where(mask==0,2,-2)



# start level set evolution

phi = drlse.drlse(img_gray.astype(np.float32), phi.astype(np.float32), W, H, mu, timestep, lmda, alfa, epsilon, 1,
                  max_iter,5,0.4)



## save the end status image

kernel = np.ones((15, 15), np.uint8)
phi_mat_binary = np.where(phi < 0, 255, 0).astype(np.uint8)


# the first results by the drlse
ours_maskarr=phi_mat_binary
dpi = 96
fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
axes = fig.add_axes([0, 0, 1, 1])
axes.set_axis_off()
axes.imshow(img_color)
axes1 = fig.add_axes([0, 0, 1, 1])
axes1.set_axis_off()
mask_rgba2 = np.ones(([H, W, 4]))
mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
mask_rgba2[:, :, 1:3] = 0
alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
mask_rgba2[:, :, -1] = alpha
axes1.imshow(mask_rgba2)
fig.savefig(os.path.join(save_dir,prename+'_drlse.jpg'))


closing = cv2.morphologyEx(phi_mat_binary, cv2.MORPH_CLOSE, kernel)

filtered_closing=closing*filter

#  the results by the morphological methods
ours_maskarr=filtered_closing.astype(np.uint8)
dpi = 96
fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
axes = fig.add_axes([0, 0, 1, 1])
axes.set_axis_off()
axes.imshow(img_color)
axes1 = fig.add_axes([0, 0, 1, 1])
axes1.set_axis_off()
mask_rgba2 = np.ones(([H, W, 4]))
mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
mask_rgba2[:, :, 1:3] = 0
alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
mask_rgba2[:, :, -1] = alpha
axes1.imshow(mask_rgba2)
fig.savefig(os.path.join(save_dir,prename+'_morph.jpg'))

'''
=======================================================================
    SLIC
=======================================================================
'''
img_lab = color.rgb2lab(img)
[H, W, Ch] = img_color.shape
data = np.reshape(img_color, (H * W, Ch))

slic_img = slic(img_color, n_segments=1000, compactness=15, sigma=0.5)

merge_location = np.where(filtered_closing != 0)

merge_label=slic_img[merge_location]

merge_class_list=np.unique(merge_label)

slic_class_list=np.unique(slic_img)

# '''
# ====================================================================================
#     computhe the union part of slic results and the segmentation meraging result,
#     when the union is over half of the number of the slic superpixels
# ==================================================================================
# '''


'''------- directly compute the union part----------'''
ravel_slic_img=np.ravel(slic_img)
final_merge_img=ravel_slic_img.copy().astype(np.uint8)
for _idx,_val in enumerate (ravel_slic_img):
    if _val in merge_class_list:
        final_merge_img[_idx]=255
    else:
        final_merge_img[_idx]=0
final_merge_img=final_merge_img.reshape(slic_img.shape)



# the segmentation results over the slic results
ours_maskarr=filtered_closing.astype(np.uint8)
dpi = 96
fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）

#show image
axes = fig.add_axes([0, 0, 1, 1])
axes.set_axis_off()
axes.imshow(img_color)

#draw the outline
contours, hierarchy = cv2.findContours(final_merge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_arr=contours[0].squeeze()
plt.plot(contour_arr[:,0],contour_arr[:,1],'cyan',linewidth=0.3)

#draw the mask
axes2 = fig.add_axes([0, 0, 1, 1])
axes1.set_axis_off()
mask_rgba2 = np.ones(([H, W, 4]))
mask_rgba2[:, :, 0] = final_merge_img / 255.*0.75 # R
mask_rgba2[:, :, 1] = final_merge_img / 255.*0.75# G
mask_rgba2[:, :, 2] = final_merge_img / 255.*0.75 # B
alpha = np.where(final_merge_img != 0, 0.9, 0) # not transparent
mask_rgba2[:, :, -1] = alpha
axes2.imshow(mask_rgba2)

#original area
axes1 = fig.add_axes([0, 0, 1, 1])
axes1.set_axis_off()
mask_rgba2 = np.ones(([H, W, 4]))
mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
mask_rgba2[:, :, 1:3] = 0
alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
mask_rgba2[:, :, -1] = alpha
axes1.imshow(mask_rgba2)

#draw the grid

axes = fig.add_axes([0, 0, 1, 1])
axes.set_axis_off()
boundaries = find_boundaries(slic_img, mode='subpixel',
                             background=0)
coors=np.where(boundaries==True)
boundary_arr=np.zeros((H,W))
boundary_arr[((coors[0])/2).astype(np.int),((coors[1])/2).astype(np.int)]=255
mask_rgba2 = np.ones(([H, W, 4]))
mask_rgba2[:, :, 0] = boundary_arr / 255.*255/255. # R
mask_rgba2[:, :, 1] = boundary_arr / 255.*238/255.# G
mask_rgba2[:, :, 2] = boundary_arr / 255.*88/255. # B

alpha = np.where(boundary_arr != 0, 0.8, 0) # not transparent
mask_rgba2[:, :, -1] = alpha
axes1.imshow(mask_rgba2)

fig.savefig(os.path.join(save_dir,prename+'_fusion.jpg'))

closing_final = cv2.morphologyEx(final_merge_img, cv2.MORPH_CLOSE, kernel)
closing_final = cv2.medianBlur(closing_final, 9)

# the final results
ours_maskarr=closing_final.astype(np.uint8)
dpi = 96
fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
axes = fig.add_axes([0, 0, 1, 1])
axes.set_axis_off()
axes.imshow(img_color)
axes1 = fig.add_axes([0, 0, 1, 1])
axes1.set_axis_off()
mask_rgba2 = np.ones(([H, W, 4]))
mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
mask_rgba2[:, :, 1:3] = 0
alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
mask_rgba2[:, :, -1] = alpha
axes1.imshow(mask_rgba2)
fig.savefig(os.path.join(save_dir,prename+'_final.jpg'))

# # show the exp results
# # the final results
#
# prename='216780_216781-100096_100097-18'
# img=Image.open(os.path.join(img_dir, prename+'.jpg'))
# img_color=np.array(img)
#
# ours_maskarr=np.array(Image.open(os.path.join(r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\mahal', prename+'.png')))
# dpi = 96
# fig2 = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
# axes = fig2.add_axes([0, 0, 1, 1])
# axes.set_axis_off()
# axes.imshow(img_color)
# axes1 = fig2.add_axes([0, 0, 1, 1])
# axes1.set_axis_off()
# mask_rgba2 = np.ones(([H, W, 4]))
# mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
# mask_rgba2[:, :, 1:3] = 0
# alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
# mask_rgba2[:, :, -1] = alpha
# axes1.imshow(mask_rgba2)
# fig2.savefig(prename+'a.jpg')
#
# ours_maskarr=np.array(Image.open(os.path.join(r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\cv_road', prename+'.png')))
# dpi = 96
# fig1 = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
# axes = fig1.add_axes([0, 0, 1, 1])
# axes.set_axis_off()
# axes.imshow(img_color)
# axes1 = fig1.add_axes([0, 0, 1, 1])
# axes1.set_axis_off()
# mask_rgba2 = np.ones(([H, W, 4]))
# mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
# mask_rgba2[:, :, 1:3] = 0
# alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
# mask_rgba2[:, :, -1] = alpha
# axes1.imshow(mask_rgba2)
# fig1.savefig(prename+'b.jpg')
#
#
#
# ours_maskarr=np.array(Image.open(os.path.join(r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\ours', prename+'.png')))
# dpi = 96
# fig3 = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
# axes = fig3.add_axes([0, 0, 1, 1])
# axes.set_axis_off()
# axes.imshow(img_color)
# axes1 = fig3.add_axes([0, 0, 1, 1])
# axes1.set_axis_off()
# mask_rgba2 = np.ones(([H, W, 4]))
# mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
# mask_rgba2[:, :, 1:3] = 0
# alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
# mask_rgba2[:, :, -1] = alpha
# axes1.imshow(mask_rgba2)
# fig3.savefig(prename+'c.jpg')
#
# ours_maskarr=np.array(Image.open(os.path.join(r'C:\python_pycharm_label_test\experiment_data\road_segmentation_reference_results_512', prename+'.png')))
# dpi = 96
# fig4 = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # 按原图像尺寸输出保存图像（尺寸和dpi）
# axes = fig4.add_axes([0, 0, 1, 1])
# axes.set_axis_off()
# axes.imshow(img_color)
# axes1 = fig4.add_axes([0, 0, 1, 1])
# axes1.set_axis_off()
# mask_rgba2 = np.ones(([H, W, 4]))
# mask_rgba2[:, :, 0] = ours_maskarr / 255. # red
# mask_rgba2[:, :, 1:3] = 0
# alpha = np.where(ours_maskarr != 0, 0.4, 0) # not transparent
# mask_rgba2[:, :, -1] = alpha
# axes1.imshow(mask_rgba2)
# fig4.savefig(prename+'d.jpg')