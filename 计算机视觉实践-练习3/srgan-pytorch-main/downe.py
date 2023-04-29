import cv2
import os


path = "E:\surreal"

for file in os.listdir(path):
    if file.endswith('png'):
        file1 = os.path.join(path,file)
        print(file)
        img = cv2.imread(file1)
        img = cv2.resize(img,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(path,'down',file),img)