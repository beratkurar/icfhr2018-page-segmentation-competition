# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:36:47 2018

@author: B
"""
import cv2
import os
os.environ["THEANO_FLAGS"] = "device=cuda0,floatX=float32"
import numpy as np
import sys
sys.path.append('/root/icfhr2018psc/testing/Models/')
np.random.seed(123)
import argparse
import Models 
from keras import optimizers

parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", type=int, default = 4 )
parser.add_argument("--input_height", type=int , default = 800  )
parser.add_argument("--input_width", type=int , default = 800 )
parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--optimizer_name", type = str , default = "sgd")
args = parser.parse_args()

n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width

optimizer_name = args.optimizer_name
model_name = args.model_name

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
sgd = optimizers.SGD(lr=0.001)

m.compile(loss='categorical_crossentropy',
      optimizer= sgd,
      metrics=['accuracy'])

print('Test pages are: ')
pagecounter=1
for page in os.listdir('Originals/'):
    print (str(pagecounter)+'-'+page)
    pagecounter=pagecounter+1

print("Loading weights")
m.load_weights('bestweightstrain98')

output_height = m.outputHeight
output_width = m.outputWidth

colors=[(255,255,255),(255,0,0),(0,0,255),(0,0,0)]

outersize=800
trimsize=200
innersize=outersize-2*trimsize

def getImageArr( img , width , height , imgNorm="divide" , odering='channels_first' ):
    try:
        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img = img/255.0
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
            return img
    except Exception as e:
        print (path)
        print (e)
        img = np.zeros((  height , width  , 3 ))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img

def predict(img):
    X = getImageArr(img , args.input_width  , args.input_height  )
    pr = m.predict( np.array([X]) )[0]
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
    seg_img = np.zeros ( (output_height , output_width , 3 ) )
    for c in range(n_classes):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    seg_img = cv2.resize(seg_img  , (input_width , input_height ))
    return seg_img

def slide(page):
    rows,cols,ch=page.shape
    x=rows//innersize
    y=cols//innersize
    prows=(x+1)*innersize+2*trimsize
    pcols=(y+1)*innersize+2*trimsize
    ppage=np.zeros([prows,pcols,3])
    ppage[trimsize:rows+trimsize,trimsize:cols+trimsize,:]=page[:,:,:]
    pred=np.ones([rows,cols,3])*255
    for i in range(0,prows-outersize,innersize):
        for j in range(0,pcols-outersize,innersize):
            patch=ppage[i:i+outersize,j:j+outersize,:]
            ppatch=predict(patch)
            pred[i:i+innersize,j:j+innersize,:]=ppatch[trimsize:trimsize+innersize,trimsize:trimsize+innersize,:]
    return pred

def convert(img):
    img=255-img
    x,y=img.shape
    criteria=(x*y)/400
    tx=x//14
    ty=y//14
    img[0:tx, :]=0
    img[-tx:, :]=0
    img[:, 0:ty]=0
    img[:, -ty:]=0
    connectivity = 4  
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    stats = output[2]
    labels = output[1]
    c=0
    for stat in stats:
        if(c!=0 and stats[c,2]>(0.8*x) or stats[c,3]>(0.8*y)):
            comp=np.array(labels==c,dtype='uint8')*255
            img=img-comp
        elif(c!=0 and stats[c,4]>criteria and (y-stats[c,0]<1000)):
            comp=np.array(labels==c,dtype='uint8')*255
            img=img-comp
        elif(c!=0 and stats[c,4]>criteria):
            comp=np.array(labels==c,dtype='uint8')*255
            _, contours, hierarchy = cv2.findContours(comp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            cont=np.zeros((x,y),dtype='uint8')
            cv2.drawContours(cont, [cnt], -1, (255,255,255), 1)
            img=img-comp+cont
        c=c+1
    return 255-img

def grey(img):
    img[np.where(img==255)]=255 #white
    img[np.where(img==76)]=128 #red
    img[np.where(img==29)]=64 #blue
    img[np.where(img==0)]=0 #black
    return img

def colormask(img,c):
    x,y=img.shape
    cimg=np.zeros((x,y),dtype='uint8')
    cimg[np.where(img==c)]=255
    return cimg

def fill(img):
    cimg=img.copy()
    x,y=img.shape
    mask=np.zeros((x+2,y+2),np.uint8)
    cv2.floodFill(cimg,mask,(0,0),255)
    img_inv=cv2.bitwise_not(cimg)
    result=img + img_inv
    return result    

def makered(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 900000
    img2 = img.copy()
    for i in range(0, nb_components):
        if sizes[i] <= min_size:
            img2[output == i + 1] = 128
    return img2

def removecolor(img,size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = size
    img2 = img.copy()
    for i in range(0, nb_components):
        if sizes[i] <= min_size:
            img2[output == i + 1] = 0
    return img2


def post1color1(black,blue,red):
    x,y=black.shape
    img = np.ones((x,y),np.uint8)*255
    img[blue==255]=64
    img[red==255]=128
    img[black==255]=0
    img[black==128]=128
    return img

def post1color2(black,blue,red):
    x,y=black.shape
    img = np.ones((x,y),np.uint8)*255
    img[blue==255]=64
    img[red==255]=128
    img[black==255]=0
    
    return img

def binary(img):
    img[img<250]=0
    return np.bitwise_not(img)
   
def removespurious(img):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    return mask    
def post1(img):
    gimg=grey(img)
    
    black=colormask(gimg,0)
    rblack=removespurious(black)
    rrblack=makered(rblack)
    
    blue=colormask(gimg,64)
    rblue=removespurious(blue)
    rrblue=removecolor(rblue,80000)
    
    red=colormask(gimg,128)
    rred=removespurious(red)
    rrred=removecolor(rred,80000)
   
    cimg=post1color1(rrblack,rrblue,rrred)
    
    black2=colormask(cimg,0)
    fblack=fill(black2)
    
    blue2=colormask(cimg,64)
    
    red2=colormask(cimg,128) 
     
    result=post1color2(fblack,blue2,red2)
    return result

def makeblue(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 200000
    img2 = img.copy()
    for i in range(0, nb_components):
        if sizes[i] <= min_size:
            img2[output == i + 1] = 64
    return img2

def post2color1(black,blue,red):
    x,y=black.shape
    img = np.ones((x,y),np.uint8)*255
    img[black==255]=0
    img[black==64]=64
    img[blue==255]=64
    img[blue==64]=64
    img[red==255]=128
    img[red==64]=64
    return img

def post2color2(black,blue,red):
    x,y=black.shape
    img = np.ones((x,y),np.uint8)*255
    img[red==255]=128
    img[red==64]=64
    img[blue==255]=64
    img[black==255]=0
    return img

def morp(img):
    kernel = np.ones((110,110),np.uint8)
    dimg = cv2.dilate(img,kernel,iterations = 6)
    return dimg
kernel = np.ones((50,50),np.uint8)
def post2(img):
    blue=colormask(img,64)
    black=colormask(img,0)
    red=colormask(img,128)
    bred=makeblue(red)
    morpblue=morp(blue)
    result=post2color2(black,morpblue,bred)
    blue2=colormask(result,64)
    rblue2=removecolor(blue2,150000)
    erblue2 = cv2.erode(rblue2,kernel,iterations = 1)
    black2=colormask(result,0)
    red2=colormask(result,128)
    rred2=removecolor(red2,200000)
    result2=post2color2(black2,erblue2,rred2)
    return result2


print('Create SegmentedPages folder if does not exist')
if not (os.path.exists('SegmentedPages')):
    os.mkdir('SegmentedPages')
print('Create Paragraphs folder if does not exist')
if not (os.path.exists('Paragraphs')):
    os.mkdir('Paragraphs')
print(' ')

predcounter=1  
for path in os.listdir('Originals'):
    print(str(predcounter) + '- Segmenting the page: '+path)
    page=cv2.imread('Originals/'+path,0)
    ret,binary_page = cv2.threshold(page,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    preprocessed_page=convert(binary_page)
    rgb_page = cv2.cvtColor(preprocessed_page,cv2.COLOR_GRAY2RGB)
    pred=slide(rgb_page)
    grey_pred= cv2.cvtColor(np.uint8(pred), cv2.COLOR_BGR2GRAY)
    post1_pred=post1(grey_pred)
    post2_pred=post2(post1_pred)
    cv2.imwrite('SegmentedPages/'+path,post2_pred)
    print(path+' has been segmented and written in SegmentedPages folder')
    print('Extracting the paragraph of '+path+' for text line segmentation')
    x,y=preprocessed_page.shape
    paragraph_mask=np.zeros((x,y),dtype='uint8')
    paragraph_mask[post2_pred==0]=1
    paragraph=preprocessed_page*paragraph_mask
    paragraph[paragraph_mask==0]=255
    rect = cv2.boundingRect(paragraph_mask)
    cropped_paragraph=paragraph[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]    
    cv2.imwrite('Paragraphs/'+path,cropped_paragraph)
    print('Paragraph of '+path+' has been extracted and written in Paragraphs folder')
    print(' ')
    predcounter=predcounter+1

print('All pages has been segmented')
