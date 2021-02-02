from shapely.geometry import LineString,shape
from PIL import ImageFilter, Image
import datetime
# import scipy.ndimage.filters as filters
import scipy.ndimage as nd
from skimage import measure, draw, morphology
import numpy as np
from skimage.color import rgb2hsv
import time
from utils import *
from scipy import stats, linalg
from compute_geodesic import get_geodesic_path, normalize_im
import cv2 as cv
from scipy import signal
import sknw
from rasterio.features import rasterize
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from drlse_tools import *
# from sklearn.cluster import MeanShift, estimate_bandwidth
# 初始种子点[153,504] [510,245]

from shapely.geometry import Point, Polygon


# from shapely.geometry import LineString
img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\images'
exp_resluts_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response'
img_list=os.listdir(img_dir)
expriment_results_img_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\predictions'
road_segmentaion_reference_img_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\GT_mask'

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

with open(os.path.join(exp_resluts_dir,'experiment_result.txt'), 'a') as f:
    f.write("\n\n")
    f.write( datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'\n')   # print the current time
    f.write("\n\n")
    f.write("item        precision   recall   F1   IoU\n" )


    Precision_cv=[]
    Recall_cv=[]
    F1_cv=[]
    IoU_cv=[]

    Precision_ours=[]
    Recall_ours=[]
    F1_ours=[]
    IoU_ours=[]


    for img_name in img_list:

        prename = img_name.split('.')[0]
        print(prename)

        road_segmentation_cv = np.array(Image.open(os.path.join(expriment_results_img_dir,'cv_road' ,prename + '.png')))
        road_segmentation_ours=np.array(Image.open(os.path.join(expriment_results_img_dir,'ours', prename + '.png')))

        road_reference_img=np.array(Image.open(os.path.join(road_segmentaion_reference_img_dir, prename + '.png')))

        _Precision_cv, _Recall_cv, _F1_cv, _IoU_cv=valuate(road_segmentation_cv, road_reference_img)
        _Precision_ours, _Recall_ours, _F1_ours, _IoU_ours = valuate(road_segmentation_ours, road_reference_img)


        Precision_cv.append(_Precision_cv)
        Recall_cv.append(_Recall_cv)
        F1_cv.append(_F1_cv)
        IoU_cv.append(_IoU_cv)

        Precision_ours.append(_Precision_ours)
        Recall_ours.append(_Recall_ours)
        F1_ours.append(_F1_ours)
        IoU_ours.append(_IoU_ours)

        f.write("%s\n" %img_name)


        f.write("CV        %.4f        %.4f        %.4f     %.4f\n" %(_Precision_cv, _Recall_cv, _F1_cv, _IoU_cv))

        f.write("ours        %.4f        %.4f        %.4f     %.4f\n" %(_Precision_ours, _Recall_ours, _F1_ours, _IoU_ours))

    f.write("average\n" )

    f.write("CV        %.4f        %.4f        %.4f     %.4f\n" % (np.average(Precision_cv), np.average(Recall_cv), np.average(F1_cv), np.average(IoU_cv)))

    f.write(
        "ours        %.4f        %.4f        %.4f     %.4f\n" % (np.average(Precision_ours),np.average(Recall_ours),np.average(F1_ours), np.average(IoU_ours)))













