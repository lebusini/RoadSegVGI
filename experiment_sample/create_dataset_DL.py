# create the dataset for Deeplearning
# including the training set, validating set and testing set
# the mask is in rgb 8-bit format

import os
import shutil
import numpy as np
import cv2
from random import shuffle
from PIL import Image




src_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample'

gt_dir=r'C:\python_pycharm_label_test\experiment_data\road_segmentation_reference_results_512'

dest_dir=os.path.join(src_dir,'segmentation_dataset_DL')

mapbox_img_dir=os.path.join(src_dir,'mapbox')

google_img_dir=os.path.join(src_dir,'google')

yahoo_img_dir=os.path.join(src_dir,'yahoo')

gts_dir=os.path.join(src_dir,'GTs')

imgs_dir=os.path.join(src_dir,'images')


#----1. move the all the images and gts to dirs of 'images' and 'GTs', and covert images to PNG and gts to [0,1]

filelist=os.listdir(mapbox_img_dir)

for file in filelist:
    prename=file.split('.')[0]

    #==== images ========

    # mapbox

    im_mp = Image.open(os.path.join(mapbox_img_dir,file))

    im_mp.save(os.path.join(imgs_dir,prename+'_mapbox.png'))

    # google

    im_gl = Image.open(os.path.join(google_img_dir,file))

    im_gl.save(os.path.join(imgs_dir,prename+'_google.png'))

    # yahoo

    im_yh = Image.open(os.path.join(yahoo_img_dir,file))

    im_yh.save(os.path.join(imgs_dir,prename+'_yahoo.png'))

    #==== GTs ========

    # mapbox

    gt_mp_arr = np.array(Image.open(os.path.join(gt_dir, prename + '.png')))

    gt_mp = Image.fromarray(np.where(gt_mp_arr == 0, 1, 0).astype(np.uint8))

    gt_mp.save(os.path.join(gts_dir, prename + '_mapbox.png'))

    # google

    gt_gl_arr = np.array(Image.open(os.path.join(gt_dir, prename + '.png')))

    gt_gl = Image.fromarray(np.where(gt_gl_arr == 0, 1, 0).astype(np.uint8))

    gt_gl.save(os.path.join(gts_dir, prename + '_google.png'))

    # yahoo

    gt_yh_arr = np.array(Image.open(os.path.join(gt_dir,prename+'.png')))

    gt_yh = Image.fromarray(np.where(gt_yh_arr==0,1,0).astype(np.uint8))

    gt_yh.save(os.path.join(gts_dir,prename+'_yahoo.png'))


#----2. select the images randomly from the "images", copy images to the "test", "train" 
# and "val" dirs, and copy gts  to the "testannot", "trainannot" and "valannot" dirs.

all_img_list=os.listdir(imgs_dir)

shuffle(all_img_list)

# 60, 20, 20 grounp

#train
length=len(all_img_list)
mid1=int(length*0.6)
mid2=int(length*0.8)

for _file in all_img_list[:mid1]:

    shutil.copyfile(os.path.join(imgs_dir,_file), os.path.join(dest_dir,'train',_file))
    
    shutil.copyfile(os.path.join(gts_dir,_file), os.path.join(dest_dir,'trainannot',_file))

#val

for _file in all_img_list[mid1:mid2]:

    shutil.copyfile(os.path.join(imgs_dir,_file), os.path.join(dest_dir,'val',_file))
    
    shutil.copyfile(os.path.join(gts_dir,_file), os.path.join(dest_dir,'valannot',_file))

#test

for _file in all_img_list[mid2:]:

    shutil.copyfile(os.path.join(imgs_dir,_file), os.path.join(dest_dir,'test',_file))
    
    shutil.copyfile(os.path.join(gts_dir,_file), os.path.join(dest_dir,'testannot',_file))