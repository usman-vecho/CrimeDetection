import cv2,os


data_dir = 'Dataset/'
categories = [i for i in os.listdir('Dataset/')]
for category in categories:
    path = os.path.join(data_dir,category)
    c = 0
    for img_ in os.listdir(path):
        img = cv2.imread(os.path.join(path,img_))
        cv2.imwrite(path+'/'+'{}.jpg'.format(c),img)
        os.remove(path+'/'+img_)
        print(path)
        c+=1

