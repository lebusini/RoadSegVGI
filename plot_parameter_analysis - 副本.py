import matplotlib.pyplot as plt
import numpy as np


def draw_polyline(nparray,savename,x_label,y_label,title_name):
    x=nparray[:,0]
    y1=nparray[:,1]
    y2=nparray[:,2]
    y3=nparray[:,3]
    plt.plot(x,y1,label='Completeness',linewidth=1,marker='.',color='r')
    plt.plot(x,y2,label='Correctness',linewidth=1,marker='.',color='g')
    plt.plot(x,y3,label='Quality',linewidth=1,marker='.',color='b')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_name)
    plt.savefig(savename+'.png')
    plt.close()

# mu
mu_test_array=np.load('mu.npy')
draw_polyline(mu_test_array,'mu_polyline.png',r'$\mu$','Score','values of metrics as the change of '+r'$\mu$')

# alpha
lmda_test_array=np.load('lmda.npy')
draw_polyline(lmda_test_array,'lmda_polyline.png',r'$\lambda$','Score','values of metrics as the change of '+r'$\lambda$')

#lamda
alfa_test_array=np.load('alfa.npy')
draw_polyline(alfa_test_array,'alfa_polyline.png',r'$\alpha$','Score','values of metrics as the change of '+r'$\alpha$')

#seg_num
n_segments_test_array=np.load('n_segments.npy')
draw_polyline(n_segments_test_array,'K_polyline.png','K','Score','values of metrics as the change of K')