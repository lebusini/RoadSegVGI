import datetime

from PIL import Image

# import scipy.ndimage.filters as filters
from utils.drlse_tools import *

# from sklearn.cluster import MeanShift, estimate_bandwidth
# 初始种子点[153,504] [510,245]

buffer_width=4

# from shapely.geometry import LineString
img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\already'
phi_binary_result_dir=r'C:\python_pycharm_label_test\experiment_data\phi_binary_results_512'
exp_resluts_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512'
img_list=os.listdir(img_dir)
phi_reference_512=r'C:\python_pycharm_label_test\experiment_data\phi_reference_results_512'
phi_skeleton_result_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts'
phi_thinned_result_dir=r'C:\python_pycharm_label_test\experiment_data\phi_thinned_results_512'
GT_dir= r'C:\python_pycharm_label_test\ground_truth'
label_single_dir=r'C:\python_pycharm_label_test\experiment_data\lebel_single_line_512'
# expriment_results_img_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\google'
# expriment_results_img_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\mapbox'
expriment_results_img_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\yahoo'
road_segmentaion_reference_img_dir=r'C:\python_pycharm_label_test\experiment_data\road_segmentation_reference_results_512'

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

    Precision_srm=[]
    Recall_srm=[]
    F1_srm=[]
    IoU_srm=[]

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

        # road_segmentation_mahal=np.array(Image.open(os.path.join(expriment_results_img_dir,'mahal' ,prename + '.png')))
        road_segmentation_srm = np.array(
            Image.open(os.path.join(expriment_results_img_dir, 'SRM', prename + '.png')))
        road_segmentation_cv = np.array(Image.open(os.path.join(expriment_results_img_dir,'cv_road' ,prename + '.png')))
        # road_segmentation_threeFeatures=np.array(Image.open(os.path.join(expriment_results_img_dir,'three_features', prename + '.png')))
        road_segmentation_ours=np.array(Image.open(os.path.join(expriment_results_img_dir,'ours', prename + '.png')))
        road_reference_img=np.array(Image.open(os.path.join(road_segmentaion_reference_img_dir, prename + '.png')))


        # Completeness_mahal, Correctness_mahal, Quality_mahal = valuate(road_segmentation_mahal, road_reference_img)
        _Precision_srm, _Recall_srm, _F1_srm, _IoU_srm = valuate(road_segmentation_srm, road_reference_img)
        _Precision_cv, _Recall_cv, _F1_cv, _IoU_cv=valuate(road_segmentation_cv, road_reference_img)

        # Completeness_threeFeatures, Correctness_threeFeatures, Quality_threeFeatures = valuate(road_segmentation_threeFeatures, road_reference_img)
        _Precision_ours, _Recall_ours, _F1_ours, _IoU_ours = valuate(road_segmentation_ours, road_reference_img)


        Precision_srm.append(_Precision_srm)
        Recall_srm.append(_Recall_srm)
        F1_srm.append(_F1_srm)
        IoU_srm.append(_IoU_srm)

        Precision_cv.append(_Precision_cv)
        Recall_cv.append(_Recall_cv)
        F1_cv.append(_F1_cv)
        IoU_cv.append(_IoU_cv)

        Precision_ours.append(_Precision_ours)
        Recall_ours.append(_Recall_ours)
        F1_ours.append(_F1_ours)
        IoU_ours.append(_IoU_ours)

        f.write("%s\n" %img_name)

        f.write("SRM        %.4f        %.4f        %.4f   %.4f \n" %(_Precision_srm, _Recall_srm, _F1_srm, _IoU_srm))

        f.write("CV        %.4f        %.4f        %.4f     %.4f\n" %(_Precision_cv, _Recall_cv, _F1_cv, _IoU_cv))

        f.write("ours        %.4f        %.4f        %.4f     %.4f\n" %(_Precision_ours, _Recall_ours, _F1_ours, _IoU_ours))

    f.write("average\n" )
    f.write("SRM        %.4f        %.4f        %.4f   %.4f \n" % (np.average(Precision_srm), np.average(Recall_srm), np.average(F1_srm), np.average(IoU_srm)))

    f.write("CV        %.4f        %.4f        %.4f     %.4f\n" % (np.average(Precision_cv), np.average(Recall_cv), np.average(F1_cv), np.average(IoU_cv)))

    f.write(
        "ours        %.4f        %.4f        %.4f     %.4f\n" % (np.average(Precision_ours),np.average(Recall_ours),np.average(F1_ours), np.average(IoU_ours)))













