from PIL import ImageFilter, Image
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology


img_dir = os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\images')

seg_results_dir=os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\predictions')



reference_dir=r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\GT_mask'
save_dir=os.path.join(r'C:\python_pycharm_label_test\compared_experiments\segmentation\extra_exp_for_response\display')


img_list=os.listdir(img_dir)
W = H = 512
dpi = 96
fig4 = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
for img_name in img_list:
    prename = img_name.split('.')[0]
    img = Image.open(os.path.join(img_dir, img_name))
    img_color = np.array(img)
    seg_img_cv=np.array(Image.open(os.path.join(seg_results_dir,'cv_road', prename+'.png')))
    seg_img_ours = np.array(Image.open(os.path.join(seg_results_dir,'ours', prename+'.png')))
    GT_img = np.array(Image.open(os.path.join(reference_dir, prename + '.png')))

    ax4 = fig4.add_axes([0, 0, 1, 1])
    ax4.set_axis_off()
    ax4.imshow(img_color)

    mask_rgba = np.ones(([H, W, 4]))
 
    #
    mask_rgba[:, :, 0] = seg_img_ours/255.
    mask_rgba[:, :, 1:3] = 0
    alpha = np.where(seg_img_ours != 0, 0.4, 0)
    mask_rgba[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba)
    fig4.savefig(os.path.join(save_dir,prename+'_ours'+'.png'))

    plt.cla()

    ax4.imshow(img_color)
    mask_rgba[:, :, 0] = seg_img_cv/255.
    mask_rgba[:, :, 1:3] = 0
    alpha = np.where(seg_img_cv != 0, 0.4, 0)
    mask_rgba[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba)
    fig4.savefig(os.path.join(save_dir,prename+'_cv' +'.png'))


    plt.cla()
    ax4.imshow(img_color)
    mask_rgba[:, :, 0] = GT_img/255.
    mask_rgba[:, :, 1:3] = 0
    alpha = np.where(GT_img != 0, 0.4, 0)
    mask_rgba[:, :, -1] = alpha
    fig4.set_size_inches(W / dpi, H / dpi)
    ax4.imshow(mask_rgba)
    fig4.savefig(os.path.join(save_dir,prename+'_GT'+'.png'))
plt.close()