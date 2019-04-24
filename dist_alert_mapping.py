# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:54:44 2019

This is a script to plot the relationship between RDoT (relative distance over 
time) and dynamic alert level. The script is designed for viewnyx project.Note 
that all the fixed values below should be tested before any practical usage.

basic setups:
    two things are considers in this simple model, a camera and an object 
    vehicle. let x1 and x2 be the absolute shift of the camera and object from
    the world cordinate, thus, x=x2-x1 is the distance between cam and obj.
    let t be the latency time of capturing current frame and last frame.
    
    we use pseudo-distance over time as the acceleration of both two vehicles 
    are keep changing time to time, please have experiments before applying 
    this method to raise dynamic alert.
    
@author: Wen Wen
"""

import numpy as np
import matplotlib.pyplot as plt

def getcolor(predict):
    enum=['g','y','r'] # for low, medium and high danger lvl
    if predict <0:
        return enum[2]
    elif predict >8:
        return enum[0]
    else:
        return enum[1]

endtime=4
timeahead=1.5
delta_t=0.1 # 10 fps if delta_t=0.1
t=np.arange(0.0,endtime,delta_t) #time, unit in second
# function for deviding lvl 0, lvl 1 and lvl 2 (low, medium and high danger)
base=1.2
f_0=30-np.log(t)/np.log(base)
f_01=20-np.log(t)/np.log(base)
f_12=10-np.log(t)/np.log(base)
f_2=5-np.log(t)/np.log(base)

delta_t=0.1
for i in range(0,int(endtime/delta_t)):
    if i==0:# draw the first point
        plt.scatter(t[i],f_0[i],c='g',marker='o')
        plt.scatter(t[i],f_01[i],c='g',marker='v')
        plt.scatter(t[i],f_12[i],c='g',marker='s')
        plt.scatter(t[i],f_2[i],c='g',marker='*')
    else:
        predict_0=(f_0[i]-f_0[i-1])/delta_t*timeahead+f_0[i]
        predict_01=(f_01[i]-f_01[i-1])/delta_t*timeahead+f_01[i]
        predict_12=(f_12[i]-f_12[i-1])/delta_t*timeahead+f_12[i]
        predict_2=(f_2[i]-f_2[i-1])/delta_t*timeahead+f_2[i]
        
        plt.scatter(t[i],f_0[i],c=getcolor(predict_0),marker='o')
        plt.scatter(t[i],f_01[i],c=getcolor(predict_01),marker='v')
        plt.scatter(t[i],f_12[i],c=getcolor(predict_12),marker='s')
        plt.scatter(t[i],f_2[i],c=getcolor(predict_2),marker='*')

plt.xlabel('time/s')
plt.ylabel('distance/m')
plt.title('Dynamic alert level based on RDoT')

plt.show()
