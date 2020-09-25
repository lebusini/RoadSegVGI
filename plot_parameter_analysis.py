import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker
img_dir = r'C:\python_pycharm_label_test\compared_experiments\segmentation\experiment_sample\parameter_analysis'
imglist=os.listdir(img_dir)
def draw_polyline(nparray,savename,x_label,y_label,title_name):
    x=nparray[:,0]
    y1=nparray[:,1]
    y2=nparray[:,2]
    y3=nparray[:,3]
    y4=nparray[:,4]
    plt.figure(figsize=(3.5, 2),dpi=300)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    # plt.plot(x, A, color="black", label="A algorithm", linewidth=1.5)
    # plt.plot(x, B, "k--", label="B algorithm", linewidth=1.5)
    # plt.plot(x, C, color="red", label="C algorithm", linewidth=1.5)
    # plt.plot(x, D, "r--", label="D algorithm", linewidth=1.5)

    plt.plot(x,y1,label='Precision',linewidth=0.5,marker='.',color='r',markersize=4)
    plt.plot(x,y2,label='Recall',linewidth=0.5,marker='^',color='g',markersize=4)
    plt.plot(x,y3,label='F1',linewidth=0.5,marker='3',color='b',markersize=4)
    plt.plot(x,y4,label='IoU',linewidth=0.5,marker='*',color='brown',markersize=4)

    # group_labels = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', ' dataset6', 'dataset7', 'dataset8',
    #                 'dataset9', 'dataset10']  # x轴刻度的标识
    # plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.xticks(fontproperties = 'Times New Roman', fontsize=8, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontproperties = 'Times New Roman',fontsize=8, fontweight='bold')
    plt.title(title_name, fontdict={'family' : 'Times New Roman', 'size': 8,'weight': 'bold'})  # 默认字体大小为12
    plt.xlabel(x_label, fontdict={'family' : 'Times New Roman', 'size': 8, 'weight': 'bold'})
    plt.ylabel(y_label, fontdict={'family' : 'Times New Roman', 'size': 8, 'weight': 'bold'})
    plt.xlim(x[0], x[-1])  # 设置x轴的范围
    ax.xaxis.set_ticks(x[::2])
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    # plt.ylim(0.5,1)
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 5}

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0,prop=font)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=2, fontweight='bold',fontproperties = 'Times New Roman')  # 设置图例字体的大小和粗细
    # plt.margins(x=0)
    plt.savefig(savename + '.png')
    plt.savefig(savename+'.svg', format='svg')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.close()


for img_name in imglist:
    prename=img_name.split('.')[0]

    # mu
    mu_test_array=np.load(prename+'_mu.npy')
    draw_polyline(mu_test_array,prename+'_mu_polyline',r'$\mu$','Score','values of metrics as the change of '+r'$\mu$')

    #alpha
    lmda_test_array=np.load(prename+'_lmda.npy')
    draw_polyline(lmda_test_array,prename+'_lmda_polyline',r'$\lambda$','Score','values of metrics as the change of '+r'$\lambda$')

    #lamda
    alfa_test_array=np.load(prename+'_alfa.npy')
    draw_polyline(alfa_test_array,prename+'_alfa_polyline',r'$\alpha$','Score','values of metrics as the change of '+r'$\alpha$')
    #
    # #seg_num
    # n_segments_test_array=np.load(prename+'_n_segments.npy')
    # draw_polyline(n_segments_test_array,prename+'_K_polyline','K','Score','values of metrics as the change of K')
    #
    # #kernel_size
    # kernel_test_array=np.load(prename+'_kenel.npy')
    # draw_polyline(kernel_test_array,prename+'_kernel_polyline','kernel_size','Score','values of metrics as the change of '+'kernel_size')
    #
    # #sigma
    # sigma_test_array=np.load(prename+'_sigma.npy')
    # draw_polyline(sigma_test_array,prename+'_sigma_polyline',r'$\sigma$','Score','values of metrics as the change of '+r'$\sigma$')

    # #iters
    # iters_test_array=np.load(prename+'_iters.npy')
    # draw_polyline(iters_test_array,prename+'_iters_polyline.png','iters','Score','values of metrics as the change of '+'iters')