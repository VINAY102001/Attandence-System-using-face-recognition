import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path='imagesattendance'
images=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)
def findEncodings(images):
    encodelist=[]
    for img3 in images:
        img3=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img3)[0]
        encodelist.append(encodings)
    return encodelist
def markattendance(name):
    with open('attendancemark.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            #dt=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{now}')

encodinglistknown=findEncodings(images)
#print(len(encodinglistknown))
cap=cv2.VideoCapture(0)
while True:
    success, img3 = cap.read()
    imgs=cv2.resize(img3,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    faceloc =face_recognition.face_locations(imgs)
    encodeloc=face_recognition.face_encodings(imgs,faceloc)
    for encodeface,face in zip(encodeloc,faceloc):
        matches=face_recognition.compare_faces(encodinglistknown,encodeface)
        facedis=face_recognition.face_distance(encodinglistknown,encodeface)
        #print(facedis)
        matchindex=np.argmin(facedis)
        if matches[matchindex]:
            name=classnames[matchindex]
            print(name)
            y1,x2,y2,x1=face
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img3,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img3,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img3,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)
    cv2.imshow("webcam",img3)
    cv2.waitKey(1)
