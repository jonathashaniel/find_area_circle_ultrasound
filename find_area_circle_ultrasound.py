# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:00:56 2019

@author: jonathas
"""

import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import measure
from scipy import signal
import os
import sys
import glob



import os
os.system('clear')
plt.close('all')
cv.destroyAllWindows()

# segment image with binary threshold, otsu and blur combinations
def Process_Image(img_dir):
    
    img1 = cv.imread(img_dir,0)
    
    val = 30
    blur = cv.GaussianBlur(img1,(5,5),0)
    ret,GO = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    Seg_img = cv.medianBlur(GO, 5)
    ret,Seg_img = cv.threshold(Seg_img,val,255,cv.THRESH_BINARY)
    Seg_img = cv.medianBlur(Seg_img, 5)
    ret,Seg_img = cv.threshold(Seg_img,val,255,cv.THRESH_BINARY)
    

    Seg_img = np.uint8(Seg_img)
    return Seg_img

# find circles using opencv library, you may need to check this part with show = True    
def Find_Circle(img, show = False):
    
    #Adjust these values!
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,3,5,param1=90,param2=120,minRadius=50,maxRadius=100)
    
    circles = np.uint16(np.around(circles))
    
    # form an intermediate image, before the average, of all circles found
    """"
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        
    cv.imshow('detected circles',cimg) #show every circles
    """       
    avg_cir= np.average(circles[0],0) 
    avg_cir = np.uint16(np.around(avg_cir))
    #plot circles
    if show == True:
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(img,'gray')
        cir_plot = plt.Circle((avg_cir[0],avg_cir[1]),avg_cir[2],color='b',fill=False)
        ax.add_patch(cir_plot)
        plt.scatter(avg_cir[0],avg_cir[1])
        plt.show()
        
    return avg_cir

def Dist_Euclidiana(v1, v2):
	v1, v2 = np.array(v1), np.array(v2)
	diff = v1 - v2
	quad_dist = np.dot(diff, diff)
	return math.sqrt(quad_dist)

def Create_Figure(img, circle, diameter, img_final,save = False, show = False):
    if save ==False and show == False:
        return
    
    plt.ioff()
    fig,(ax,ax2) = plt.subplots(1, 2)
    fig.suptitle('frame{}    Diameter={}'.format(i+1, diameter))
    ax.set_aspect('equal')
    ax.imshow(cv.imread(img,0),'gray')
    cir_plot = plt.Circle((circle[0],circle[1]),diameter/2,color='b',fill=False)
    ax.add_patch(cir_plot)
    ax.scatter(circle[0],circle[1])
    ax2.imshow(img_final)
    cir_plot = plt.Circle((circle[0],circle[1]),circle[2],color='r',fill=False)
    ax2.add_patch(cir_plot)
    
    if not os.path.exists(dir_path + 'fig'):
        os.mkdir(dir_path+'./fig')


    if save:
        fig.savefig('fig/teste-frame{}.png'.format(i+1))
    if show:
        fig.show()
    else:
        plt.close()   
    
    plt.ion()


def Area_Mensurement(img,circle, show = False):
    area=0
    A= []
    
    area_img_show = cv.cvtColor(img ,cv.COLOR_GRAY2RGB)
    area_img = np.uint8(np.zeros(img.shape))
    for x in range(len(img[0,:])):
        for y in range(len(img[:,0])):
            if img[y,x] == 0:
                #check if it is inside the circle
                if Dist_Euclidiana([y,x],[circle[1],circle[0]]) <= circle[2]:
                    area_img[y,x] = 1

    label_img = measure.label(area_img)
    region = measure.regionprops(label_img)
    for l in region:
        A.append(l.filled_area)
    area=max(A)
    best = A.index(area)
    lab = region[best].label
        

    for x in range(len(label_img[0,:])):
        for y in range(len(label_img[:,0])):
            if label_img[y,x] == lab:
                    area_img_show[y,x] = [255,255,0]
    if show == True:
        plt.imshow(area_img_show)
        
    return area, area_img, area_img_show

def FirstOrder_Filter(data, Wn = .2, order=1, show = False):
    b, a = signal.butter(order, Wn)
    y = signal.filtfilt(b, a, data)
    if show:
        plt.figure()
        plt.plot(data, 'b', alpha=0.75)
        plt.plot(y, 'k--')
        plt.legend(('original','filtered'), loc='best');    plt.grid(True)
        plt.show()
    return y
    



#############  CODE      #############################
Areas= []
Diameters = []
Q_frames = 109
dir_path = 'D:/Python codes/area_image/'
pixel_to_mm = 1



if not os.path.exists(dir_path):
    sys.exit('ERROR dir_path')
    
frames = [f for f in glob.glob(dir_path + "*.png", recursive=True)]
frames1 = [f.split('frame') for f in frames]

#folders = [f for f in glob.glob(dir_path + "*/", recursive=True)]


#for img in frames:
for i  in range(Q_frames):
    img = 'frame{}.png'.format(i+1)
    
    Pimg = Process_Image(img)
    circle = Find_Circle(Pimg,show = False)
    area, areaI, img_final = Area_Mensurement(Pimg,circle,show = False)
    diameter = math.sqrt(area/math.pi)*2
    Create_Figure(img, circle, diameter, img_final,save = True, show = False)
    
    Areas.append(area)
    Diameters.append(diameter)
        
    print(img[:-4])
    print('area',area)
    print('diameter', diameter)
    
    
Filtered_diameters = FirstOrder_Filter(Diameters, show = True) 
    
######################################################
    
  



    

    


