import os
import time
from random import random 
import time
import cv2
cap=cv2.VideoCapture(0)


FACE_CASCADE=cv2.CascadeClassifier('Face_cascade.xml')​

def capture(d):
    i=0
    while(True):
        _,frame=cap.read()
       image=frame.copy
    #img =cv2.resize(frame,(550,400))
        for i in range(10):
            time.sleep(0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
            
            for x,y,w,h in faces:
                    sub_img=image[y-10:y+h+10,x-10:x+w+10]
                    os.chdir("Extracted")
                    #cv2.imwrite(str(randint(0,10000))+".jpg",sub_img)
                    cv2.imwrite(d+"/ train"+str(random())+".jpg",sub_img)
                    os.chdir("../")
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)
            if i == 9:
               break
            print(i)
        if i == 9:
               break
         
            #time.sleep(1)
        #cv2.imshow("frame",frame)
        #key = cv2.waitKey(1) & 0xFF
    
    #cap.release()
    #cv2.destroyAllWindows()
​
​
​
print("Enter your ID:")
id=input().strip()
print("ID RECIVED")
path = r"C:\Users\Prakhar\Desktop\Sample_face_rec_mark1\dataset-try"
​
counter1=0
counter2=0
​
​
files=os.listdir(path)
#print(len(files))
ID_LIST=[]
​
for name in files :
    #print(name)
    #print(1)
    ID_LIST.append(name)
    #print(len(ID_LIST))
print(ID_LIST)
​
​
for i in range (len(ID_LIST)):
    a=ID_LIST[i].strip()
    b=ID_LIST[i].strip()
    print(a)
    print(len(a))
    print(len(id))
    
    if (id == a):
        print(1)
        d=path+"/"+b
        print(d)
        counter1=1
        counter2=0
        for i in range(2):
            capture(d)
    else:
        print(2)
        counter2=2
        counter1=0
    
if(counter2== 2 and counter1==0):
        d= os.path.join(path, id)
        #print(d)
        try:
            os.mkdir(d)
            for i in range(2):
                capture(d)
        except FileExistsError as e:
                print("error")
                
if (ID_LIST == []):
     d= os.path.join(path, id)
     #print(d)
     os.mkdir(d)
     for i in range(2):
         capture(d)
