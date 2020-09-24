from PIL import ImageFilter, Image
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

map_name='mapbox'
# map_name='google'
# map_name='yahoo'

# img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\already'
# img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\google'
img_dir = os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample',map_name)
# img_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\GT\img'

# seg_results_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\mapbox'
# seg_results_dir=r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts\google'
seg_results_dir=os.path.join(r'C:\python_pycharm_label_test\experiment_data\exp_results_512\compared_experiment_resluts',map_name)


reference_dir=r'C:\python_pycharm_label_test\experiment_data\road_segmentation_reference_results_512'
# reference_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\GT\GT'

# save_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\mapbox'
# save_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\google'
save_dir=os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display',map_name)
# save_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_results\display\GT\show'


img_list=['216800_216801-100114_100115-18.jpg',
'216748_216749-100134_100135-18.jpg',
'216752_216753-100166_100167-18.jpg',
'216792_216793-100178_100179-18.jpg',
'216804_216805-100080_100081-18.jpg'  ]

# img_list=os.listdir(img_dir)
W = H = 512
dpi = 96
fig4 = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
for img_name in img_list:
    prename = img_name.split('.')[0]
    img = Image.open(os.path.join(img_dir, img_name))
    img_color = np.array(img)
    seg_img_srm=np.array(Image.open(os.path.join(seg_results_dir,'SRM', prename+'.png')))
    seg_img_cv = np.array(Image.open(os.path.join(seg_results_dir,'cv_road', prename+'.png')))
    seg_img_ours = np.array(Image.open(os.path.join(seg_results_dir,'ours', prename+'.png')))
    GT_img = np.array(Image.open(os.path.join(reference_dir, prename + '.png')))

    ax4 = fig4.add_axes([0, 0, 1, 1])
    ax4.set_axis_off()
    ax4.imshow(img_color)
    mask_rgba_ours = np.ones(([H, W, 4]))
    mask_rgba_cv = np.ones(([H, W, 4]))
    mask_rgba_srm = np.ones(([H, W, 4]))
    mask_rgba_GT = np.ones(([H, W, 4]))
    #
    mask_rgba_ours[:, :, 0] = seg_img_ours/255.
    mask_rgba_ours[:, :, 1:3] = 0
    alpha = np.where(seg_img_ours != 0, 0.4, 0)
    mask_rgba_ours[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba_ours)
    fig4.savefig(os.path.join(save_dir,'ours_'+prename+'.png'))

    plt.cla()

    ax4.imshow(img_color)
    mask_rgba_cv[:, :, 0] = seg_img_cv/255.
    mask_rgba_cv[:, :, 1:3] = 0
    alpha = np.where(seg_img_cv != 0, 0.4, 0)
    mask_rgba_cv[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba_cv)
    fig4.savefig(os.path.join(save_dir,'cv_road_' +prename+'.png'))

    plt.cla()
    ax4.imshow(img_color)
    mask_rgba_srm[:, :, 0] = seg_img_srm/255.
    mask_rgba_srm[:, :, 1:3] = 0
    alpha = np.where(seg_img_srm != 0, 0.4, 0)
    mask_rgba_srm[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba_srm)
    fig4.savefig(os.path.join(save_dir,'srm_'+prename+'.png'))

    plt.cla()
    ax4.imshow(img_color)
    mask_rgba_GT[:, :, 0] = GT_img/255.
    mask_rgba_GT[:, :, 1:3] = 0
    alpha = np.where(GT_img != 0, 0.4, 0)
    mask_rgba_GT[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba_GT)
    fig4.savefig(os.path.join(save_dir,'GT_'+prename+'.png'))
plt.close()