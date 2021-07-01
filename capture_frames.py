import cv2
import numpy

c = 0
cap = cv2.VideoCapture('test_videos/firing1.mp4')
while True:
    _,frame =  cap.read()
    frame = cv2.resize(frame,(600,600))
    cv2.imwrite('del/{}.jpg'.format(str(c)),frame)
    key = cv2.waitKey(33)
    if key==27:
        cv2.destroyAllWindows()
        break
    c+=1
cap.release()
