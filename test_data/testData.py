import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import cv2, os, sys
from PIL import Image
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tfback
from keras import backend as K
K.common.set_image_dim_ordering('th')


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus



def loadModel(model_dir, model_name='final_model'):
	json_file = open('{}/{}.json'.format(model_dir, model_name), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = keras.models.model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights("{}/{}.h5".format(model_dir, model_name))

	return (loaded_model, loaded_model_json)



def predictResult(train_data, model):
	s=''
	for i in range(len(train_data)):
	    train_data[i]=np.array(train_data[i])
	    train_data[i]=train_data[i].reshape(1,1,28,28)
	    result=model.predict_classes(train_data[i])
	    if(result[0]==10):
	        s=s+'-'
	    if(result[0]==11):
	        s=s+'+'
	    if(result[0]==12):
	        s=s+'*'
	    if(result[0]==0):
	        s=s+'0'
	    if(result[0]==1):
	        s=s+'1'
	    if(result[0]==2):
	        s=s+'2'
	    if(result[0]==3):
	        s=s+'3'
	    if(result[0]==4):
	        s=s+'4'
	    if(result[0]==5):
	        s=s+'5'
	    if(result[0]==6):
	        s=s+'6'
	    if(result[0]==7):
	        s=s+'7'
	    if(result[0]==8):
	        s=s+'8'
	    if(result[0]==9):
	        s=s+'9'

	return s



def processImage(image_dir, model, model_json, filename):
	img = cv2.imread('{}'.format(image_dir),cv2.IMREAD_GRAYSCALE)
	#kernel = np.ones((3,3),np.uint8)
	cv2.imshow("wo",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#erosion = cv2.erode(img,kernel,iterations = 3)
	#dilation = cv2.dilate(img,kernel,iterations = 1)
	#img=dilation
	
	if img is not None:
	    #images.append(img)
	    img=~img
	    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	    ctrs,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	    w = int(28)
	    h = int(28)
	    train_data=[]

	    #print(len(cnt))
	    rects=[]
	    for c in cnt :
	        x,y,w,h= cv2.boundingRect(c)
	        rect=[x,y,w,h]
	        rects.append(rect)

	    #print(rects)
	    bool_rect=[]
	    for r in rects:
	        l=[]
	        for rec in rects:
	            flag=0
	            if rec!=r:
	                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
	                    flag=1
	                l.append(flag)
	            if rec==r:
	                l.append(0)
	        bool_rect.append(l)

	    #print(bool_rect)
	    dump_rect=[]
	    for i in range(0,len(cnt)):
	        for j in range(0,len(cnt)):
	            if bool_rect[i][j]==1:
	                area1=rects[i][2]*rects[i][3]
	                area2=rects[j][2]*rects[j][3]
	                if(area1==min(area1,area2)):
	                    dump_rect.append(rects[i])

	    #print(len(dump_rect)) 
	    final_rect=[i for i in rects if i not in dump_rect]
	    #print(final_rect)
	    for r in final_rect:
	        x=r[0]
	        y=r[1]
	        w=r[2]
	        h=r[3]
	        im_crop =thresh[y:y+h+10,x:x+w+10]
	        

	        im_resize = cv2.resize(im_crop, (28, 28))
	        cv2.imshow("work", im_resize)
	        cv2.waitKey(0)
	        cv2.destroyAllWindows()

	        im_resize=np.reshape(im_resize,(1, 28, 28))
	        train_data.append(im_resize)

	return train_data



def testImages(model_dir = '../models', model_name = 'final_model', image_dir = '../static/equation.jpg'): 
	model, model_json = loadModel(model_dir, model_name)
	print("Model Loaded Successfully as `{}`!!".format(model_name))

	train_data = processImage(image_dir, model, model_json, image_dir)
	# print(train_data)
	prediction = predictResult(train_data, model)

	answer = eval(prediction)

	return (prediction, answer)



def testData(model_dir = '../models', model_name = 'final_model', image_dir = '../static/equation.jpg'):
	testImages(
		model_dir = model_dir, 
		model_name = 'final_model', 
		image_dir = image_dir
	)


def main():
	testData()

if __name__ == '__main__':
	print("Using tensorflow version: "+str(tf.__version__))
	print("using keras version: "+str(keras.__version__))

	main()