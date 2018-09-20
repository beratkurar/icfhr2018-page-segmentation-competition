# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:43:28 2017

@author: B
"""

import cv2
import os

patchSize=320
patchNumber=0
folder='train/'
lfolder='ltrain/'
patch_size=800
step_size=400
c=0
for page_name in os.listdir(folder):
    page=cv2.imread(folder+page_name,0)
    lpage=cv2.imread(lfolder+page_name,0)
    rows,cols=page.shape
    for i in range(0,rows-patch_size,step_size):
        for j in range(0,cols-patch_size,step_size):
            patch=page[i:i+patch_size,j:j+patch_size]
            lpatch=lpage[i:i+patch_size,j:j+patch_size]
            cv2.imwrite("p"+folder+page_name[:-4]+"_patch"+str(c)+".png",patch)
            cv2.imwrite("p"+lfolder+page_name[:-4]+"_patch"+str(c)+".png",lpatch)
            c=c+1

