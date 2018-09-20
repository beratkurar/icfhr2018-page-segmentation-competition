# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:14:50 2017

@author: B
"""
import sys
sys.path.append('/root/icfhr2018psc/training/Models/')
import numpy as np
np.random.seed(123)
import argparse
import Models , PageLoadBatches
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str,default ="bestweights"  )
parser.add_argument("--train_images", type = str, default ="ptrain/"  )
parser.add_argument("--train_annotations", type = str, default = "pltrain/"  )
parser.add_argument("--n_classes", type=int, default = 4 )
parser.add_argument("--input_height", type=int , default = 800  )
parser.add_argument("--input_width", type=int , default = 800 )


parser.add_argument("--epochs", type = int, default = 400 )
parser.add_argument("--batch_size", type = int, default = 3 )

parser.add_argument("--load_weights", type = str , default = '')

parser.add_argument("--model_name", type = str , default = "fcn8")

args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
model_name = args.model_name

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
sgd = optimizers.SGD(lr=0.001)
#adm=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

m.compile(loss='categorical_crossentropy',
      optimizer= sgd,
      metrics=['accuracy'])

if len( load_weights ) > 0:
    print("loading initial weights")
    m.load_weights(load_weights)

colors=[(255,255,255),(255,0,0),(0,0,255),(0,0,0)]

print ( m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth
print ( output_height)
print ( output_width)
G  = PageLoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

mcp=ModelCheckpoint( filepath=save_weights_path, monitor='loss', save_best_only=True, save_weights_only=True,verbose=1)


for ep in range( epochs ):
    m.fit_generator( G , 237 , epochs=1,callbacks=[mcp] )
    '''
    print('testing')
    for page in os.listdir('btrain/'):
        img='btrain/'+page
        X = PageLoadBatches.getImageArr(img , args.input_width  , args.input_height  )
        pr = m.predict( np.array([X]) )[0]
        pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
        seg_img = np.zeros( ( output_height , output_width , 3  ) )
        for c in range(n_classes):
            seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
            rseg_img = cv2.resize(seg_img  , (input_width , input_height ))
        cv2.imwrite('predicts/'+page,rseg_img)
    '''
