from PIL import ImageFilter, Image
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology


img_dir = os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\images')

seg_results_dir_DL=os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\predictions_DL')


seg_results_dir_ours=os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\predictions')

reference_dir=r'C:\python_pycharm_label_test\experiment_data\road_segmentation_reference_results_512'
save_dir=os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\display_DL')


img_list=os.listdir(os.path.join(seg_results_dir_DL,'FPN'))
W = H = 512
dpi = 96
fig4 = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
for img_name in img_list:
    prename = img_name.split('.')[0]
    img = Image.open(os.path.join(img_dir, img_name))
    img_color = np.array(img)
    seg_img_FPN=np.array(Image.open(os.path.join(seg_results_dir_DL,'FPN', img_name)))
    seg_img_Unet = np.array(Image.open(os.path.join(seg_results_dir_DL,'Unet', img_name)))
    seg_img_ours = np.array(Image.open(os.path.join(seg_results_dir_ours, img_name)))
    ori_name='_'.join(prename.split('_')[:3])
    GT_img = np.array(Image.open(os.path.join(reference_dir, ori_name + '.png')))

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
    fig4.savefig(os.path.join(save_dir,prename+'_ours'+'.png'))

    plt.cla()

    ax4.imshow(img_color)
    mask_rgba_cv[:, :, 0] = seg_img_FPN/255.
    mask_rgba_cv[:, :, 1:3] = 0
    alpha = np.where(seg_img_FPN != 0, 0.4, 0)
    mask_rgba_cv[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba_cv)
    fig4.savefig(os.path.join(save_dir,prename+'_FPN' +'.png'))

    plt.cla()
    ax4.imshow(img_color)
    mask_rgba_srm[:, :, 0] = seg_img_Unet/255.
    mask_rgba_srm[:, :, 1:3] = 0
    alpha = np.where(seg_img_Unet != 0, 0.4, 0)
    mask_rgba_srm[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba_srm)
    fig4.savefig(os.path.join(save_dir,prename+'_Unet'+'.png'))

    plt.cla()
    ax4.imshow(img_color)
    mask_rgba_GT[:, :, 0] = GT_img/255.
    mask_rgba_GT[:, :, 1:3] = 0
    alpha = np.where(GT_img != 0, 0.4, 0)
    mask_rgba_GT[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba_GT)
    fig4.savefig(os.path.join(save_dir,prename+'_GT'+'.png'))
plt.close()