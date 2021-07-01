import random,os,pickle,cv2
import numpy as np


img_size = 224

training_data = []
data_dir = 'Dataset/'
catagories = [i for i in os.listdir(data_dir)]
def create_training_data():
    for catagory in catagories:
        class_num = catagories.index(catagory)
        
        path = os.path.join(data_dir , catagory)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array , class_num])

            except:
                pass            
create_training_data()
random.shuffle(training_data)
np.save('Dataset_v1.npy',training_data)
