import keras
import numpy as np
import cv2,os
model = keras.models.load_model('crime_model.model')
cats_ = [i for i in os.listdir('Dataset/')]

cap = cv2.VideoCapture('test_videos/bulglary.mp4')
try:
    while True:
        _,frame = cap.read()
        img = frame.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))
        actual = np.argmax(list(model.predict([img.reshape(-1,224,224,3)])))
        print(list(model.predict([img.reshape(-1,224,224,3)])),'corresponding action : ',cats_[actual])

        frame = cv2.resize(frame,(600,400))
        cv2.putText(frame,str(cats_[actual]),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))
        cv2.imshow('',frame)
        key = cv2.waitKey(33)
        if key==27:
            cv2.destroyAllWindows()
            break
    cap.release()
except:
    pass

















