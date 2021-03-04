import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from SRM import SRM
from skimage import measure
import cv2
import os
import time
import datetime
map_type= sys.argv[1]
# img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\already'
# img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\yahoo'
img_dir = os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample',map_type)
exp_resluts_dir = r'C:\python_pycharm_label_test\experiment_data\exp_results_512'
img_results_dir=os.path.join(r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts',map_type,'SRM')


img_list = os.listdir(img_dir)

with open(os.path.join(exp_resluts_dir, 'experiment_result.txt'), 'a') as f:
    f.write("\n\n")
    f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')  # print the current time
    f.write('SRM_method \n')  # print the method name
    f.write("\n\n")
    f.write("name        time   \n")
    for img_name in img_list:
        start = time.time()
        prename = img_name.split('.')[0]
        img = Image.open(os.path.join(img_dir, img_name))
        q = 700
        im=np.array(img)

        srm = SRM(im, q)
        # 1.run the SRM
        segmented = srm.run()

        # 2.threshold the image by three values mapbox (80,150) yahoo (60,110) google (70,130)
        binary_img=((segmented[:,:,0]<130) *(segmented[:,:,1]<130)*(segmented[:,:,2]<130) *(segmented[:,:,0]>69)*(segmented[:,:,1]>69)*(segmented[:,:,2]>69)*255).astype(np.uint8)

        label_region_image = measure.label(binary_img, connectivity=2)
        road_binary_regions = measure.regionprops(label_region_image)

        # 3.remove very small road region
        region_area_list=[]
        for _region in road_binary_regions:
            region_area_list.append(_region.area)
            if _region.area <20000:
                binary_img[_region.coords[:, 0], _region.coords[:, 1]] = 0

        # 4.remove the small aspect ration regions
            else:
                aspect=_region.major_axis_length/_region.minor_axis_length # use the fitted ellipse major axis and minor axis
                if aspect<1.5 and _region.area <50000:
                    binary_img[_region.coords[:, 0], _region.coords[:, 1]] = 0
        if np.sum(binary_img)==0: # If nothing is road, leave the largest region
            index_area=np.argsort(region_area_list)
            binary_img[road_binary_regions[index_area[-1]].coords[:, 0],road_binary_regions[index_area[-1]].coords[:, 1]]=255


        cv2.imwrite(os.path.join(img_results_dir, prename + '.png'), binary_img)
        end = time.time()
        dur_time = end - start
        f.write("%s        %d   \n" % (prename, dur_time))
        print(prename)
