import tensorflow as tf 
import keras 
import cv2
import copy 
import numpy as np
from tensorflow.keras.models import model_from_json


#n Model From Kaggle 

model_json_file = 'model.json'
model_weights_file = 'model_weights.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)
    
# model 
model=keras.models.load_model('55_vgg19.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotions=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


cap = cv2.VideoCapture(0)

while True:
    
    ret,frame = cap.read()
    img = copy.deepcopy(frame)
    if frame is None :
        print('--(!) No captured frame -- Break!')
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in  faces:
        fc = gray[y:y+h,x:x+w]
        
        roi = cv2.resize(fc, (48,48))
        pred = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
#   

#         img1 = cv2.resize(fc,(224,224),interpolation=cv2.INTER_LINEAR)
#         img2 = cv2.merge([img1,img1,img1])
#         batch_img = np.expand_dims(img2, axis=0)
        
        #predicitng the labels
        
        pred = model.predict(batch_img)
        
        idx = np.argmax(pred)
        text = emotions[idx]
        
        cv2.putText(img,text,(x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
        img  = cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),3)
        
        
    cv2.imshow('Emotion_detector',img)
    key = cv2.waitKey(1) & 0xFF
    if key== ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
