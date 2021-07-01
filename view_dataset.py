import pickle,os,cv2

categories = ['bulglary','fight','firing']
data = pickle.load(open('X.pickle','rb'))
labels = pickle.load(open('Y.pickle','rb'))
for i,j in zip(data,labels):
    img = i
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow('-',img)
    print('Activity :',categories[j])
    cv2.waitKey(0)
