from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from keras.models import load_model
import numpy as np
import face_recognition
import pickle
import time
import cv2
import os
import dlib
import psutil
import datetime 
import csv
import imutils
from imutils import face_utils

EYE_AR_THRESH = 0.21
RATIO_THRESH = 0.0017
EYE_AR_CONSEC_FRAMES = 3
g=0
COUNTER = 0
TOTAL = 0
FACE=0


date=str(datetime.datetime.now())
d1=str(date[0:10])
d2=d1
print(d1) ##date 
g=0
d=[]
c=0.5
f=0
ID_NAME=""
z=0
j=3
b=0
q=0
m=2
l=0 #counter for logging 
p=open("Attendance/log.txt", "r")   # loading of log.txt ... includes the last date of systems operartion 
contents =p.read()
if (contents ==''):
    l=1
if contents != d1  or l == 1:
    f= open("Attendance/log.txt","w+")   
    f.write(d1)
    f.close()
    with open('Attendance/'+str(d1)+'.csv' , 'w') as csvfile:
                attendance=csv.writer(csvfile,delimiter =',',quotechar='|', quoting =csv.QUOTE_MINIMAL)   #cvs file creation of respective date
                attendance.writerow(['Name','Time'])




encodings="employee_1.pickle"  
detection_method= "hog"
data = pickle.loads(open(encodings, "rb").read())

shape_predictor="liveliness\shape_predictor_68_face_landmarks.dat"
d=[]
c=0.5
f=0
ID_NAME=""
#le="le.pickle"
model="liveliness\liveness.model"


protoPath ="face_detector/deploy.prototxt"
modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

#loading HOG based landmark predictors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the model for liveliness

model = load_model(model)


# initialize the video stream and allow the camera sensor to warmup
print("FACE RECOGNITION ATTENDANCE SYSEM IS STARTING /\.../\.")
#cap = cv2.VideoCapture("http://192.168.43.1:8080"+"/video")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

def clock():
    t1=str(datetime.datetime.now())
    t2=t1[13:20]
    hrs=t1[11:13]
    time_=str(abs(int(hrs))-12)+str(t2)
    a=int(t1[11:13])
    if (a > 0 and a <=12):
        time_=str(abs(int(hrs)))+str(t2)+"Am"
    else:
        time_=str(abs(int(hrs))-12)+str(t2)+"PM"
    return time_



def atten_log(ID_NAME,m,time_):      #function for attendance logging in .csv file of respective date 
        with open('Attendance/'+str(d1)+'.csv' , mode='r') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=',')
            line_count = 1
            for row in csv_reader:
                a=str(row[0:1]).replace('[','').replace(']','')
                #print(a)
                if ID_NAME in a:
                    m=m+1
                    #print(m)
            if(m>2):
                return 0 #if attendance is already logged return 0
                m=0
            elif(m==2):
                    with open('Attendance/'+str(d1)+'.csv' ,'a') as csvfile2:
                           attendance=csv.writer(csvfile2)#,delimiter =',',quotechar='|', quoting =csv.QUOTE_MINIMAL)
                           attendance.writerow([ID_NAME, time_])
                           h="appended"
                           return 1  #return 1 after appending new attendance



def eye_aspect_ratio(eye):
                # to compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
                A = dist.euclidean(eye[1], eye[5])
                B = dist.euclidean(eye[2], eye[4])

                # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
                C = dist.euclidean(eye[0], eye[3])

                # compute the eye aspect ratio
                ear = (A + B) / (2.0 * C)

                # return the eye aspect ratio
                return ear



def recogniser(frame,encodings,detection_method,data,f,d,g):
    while True:
                _,frame = cap.read()                
                # convert the input frame from BGR to RGB then resize it to have
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb = cv2.resize(frame, (500,500))
                r = frame.shape[1] / float(rgb.shape[1])
                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input frame, then compute
                # the facial embeddings for each face
                boxes = face_recognition.face_locations(rgb,
                                model=detection_method)
                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []
                for encoding in encodings:
                                # attempt to match each face in the input image to our known encodings
                                matches = face_recognition.compare_faces(data["encodings"],
                                                encoding)
                                name = "Unknown"
                                #if name== "Unknown":
                                     #data_collector
                                # check to see if we have found a match
                                if True in matches:
                                                # find the indexes of all matched faces then stores in dictionary the number of time a face matches 
                                                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                                                counts = {}

                                                # loop over the matches and  count for each recognized face face
                                                for i in matchedIdxs:
                                                                name = data["names"][i]
                                                                counts[name] = counts.get(name, 0) + 1

                                                # determine the recognized face with the largest number of votes 
                                                name = max(counts, key=counts.get)
                                
                                # update the list of names
                                names.append(name)

                # loop over the recognized faces
                for ((top, right, bottom, left), name) in zip(boxes, names):
                                # scaling the face coordinates
                                top = int(top * r)
                                right = int(right * r)
                                bottom = int(bottom * r)
                                left = int(left * r)

                                # predicted face name on the image
                                cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
                                y = top - 15 if top - 15 > 15 else top + 15
                                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                                name2= name
                                
                                if name == name2:
                                    f=f+1
                                    d.append(name)
                                    
                                    if len(d) ==2:
                                        #g=g+1 
                                        if f >=1:
                                            #print(len(d))
                                            print("Person Identified :"+name2)
               
                if len(d)==2:
                    #g=g+1
                    return name2
                    break



gamma=0.1   
# loop over the frames from the video stream
while True:
                #print("CPU ussage:")
                #print(psutil.cpu_percent())
                #print("RAM usage:")
                #print(psutil.virtual_memory())
                _,frame = cap.read()
                
                
                frame = cv2.resize(frame, (600,500))
                #_,imgg=cap.read()
                #img = cv2.resize(imgg, (600,500))
                crop_image = frame[20:600, 200:3000]
                cv2.rectangle(frame,(600,25),(190,500),(0,255,0),2)
                # grab the frame dimensions and convert it to a blob
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300), interpolation = cv2.INTER_LINEAR), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #gray =cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
                #gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)*0
                
                #ind = np.int((gray2.shape[1]/3.2))
                #img[:,0:ind,:] = cv2.cvtColor(gray2[:,0:ind], cv2.COLOR_GRAY2BGR)
                
                # pass the blob through the network and obtain the detections and predictions
                net.setInput(blob)
                detections = net.forward()
                rects = detector(gray, 0)
                # loop over the detections
                if len(rects) == 1:
                    cv2.putText(frame, "", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    for i in range(0, detections.shape[2]):
                                # extract the probability associated with the prediction
                                    confidence = detections[0, 0, i, 2]

                                # remove weak detections
                                    if confidence > c:
                                                # calculate the coordinates of the bounding box for the face and extract the face ROI
                                                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                                    (X, Y, lastX, lastY) = box.astype("int")

                                                # ensure the detected bounding box does fall outside the dimensions of the frame
                                                    X = max(0, X)
                                                    Y = max(0, Y)
                                                    lastX = min(w, lastX)
                                                    lastY = min(h, lastY)

                                                # extract the face ROI and then preproces it in the exact same manner as our training data
                                                    face = frame[Y:lastY, X:lastX]
                                                    face = cv2.resize(face, (32, 32),interpolation = cv2.INTER_LINEAR)
                                                    face = face.astype("float") / 255.0
                                                    face = img_to_array(face)
                                                    face = np.expand_dims(face, axis=0)

                                                # pass the face ROI through the trained liveness detector
                                                # model to determine if the face is "real" or "fake"
                                                    preds = model.predict(face)[0]
                                                    j = np.argmax(preds)

                                                # draw the label and bounding box on the frame
                                           
                                                    if(j==0):
                                                            cv2.putText(frame, "real", (X, Y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                                            FACE=FACE+1
                                                           
                                                        #print(FACE)
                                                    if(j==1):
                                                            cv2.putText(frame, "fake", (X, Y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                                            FACE=0
                                                    cv2.rectangle(frame, (X, Y), (lastX, lastY),(255, 0, 255), 2)
                                                    for rect in rects:
                                                        cv2.putText(frame, "BLINK YOUR EYES COUPLE OF TIMES FOR FACE REALITY CHECK", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)
                                # determine the facial landmarks for the face region, then
                                # convert the facial landmark (x, y)-coordinates to a NumPy
                                # array         cv2.putText(frame, "Face is Real ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                                        shape = predictor(gray, rect)
                                                        shape = face_utils.shape_to_np(shape)

                                # extract the left and right eye coordinates, then use the
                                # coordinates to compute the eye aspect ratio for both eyes
                                                        leftEye = shape[lStart:lEnd]
                                                        rightEye = shape[rStart:rEnd]
                                                        lefteye = eye_aspect_ratio(leftEye)
                                                        righteye= eye_aspect_ratio(rightEye)

                                # average the eye aspect ratio together for both eyes
                                                        eye = (lefteye + righteye) / 2.0
                                                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                                                        eye_ratio = eye/w
                                # compute the convex hull for the left and right eye, then
                                # visualize each of the eyes
                                                        leftEyeHull = cv2.convexHull(leftEye)
                                                        rightEyeHull = cv2.convexHull(rightEye)
                                                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                                                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                                # check to see if the eye aspect ratio is below the blink
                                # threshold, and if so, increment the blink frame counter
                                                        if eye_ratio < RATIO_THRESH:
                                
                                                                COUNTER += 1

                                                        else:
                                                # if the eyes were closed for a sufficient number of
                                                # then increment the total number of blinks
                                                                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                                                            TOTAL += 1

                                                # reset the eye frame counter
                                                                            COUNTER = 0

                                
                                                           # if key ==ord("w"):
                                                                # TOTAL = 0
                                                                A="EYE BLINKS: {}".format(TOTAL)
                                                                B=int(format(TOTAL))
                                                                cv2.putText(frame, A, (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                                                if(B>4 and FACE > 1):
                                                                    cv2.putText(frame, "", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
                                                                    cv2.putText(frame, "EYE BLINK RECORDED ", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
                                                                    cv2.putText(frame, "", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                                                    cv2.putText(frame, "Face is Real ", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                                                    d=[]
                                                                    ID = recogniser(frame,encodings,detection_method,data,f,d,g)
                                                                    cv2.putText(frame, "ID:"+ID , (10, 140),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                                                    ID_NAME=ID
                                                                    t1=clock()
                                                                    v=atten_log(ID,m,t1)
                                                                    if v== 1:
                                                                        #print(v)
                                                                        z=40
                                                                    TOTAL=0
                                                                    if v==0:
                                                                        print(j)
                                                                        b=40
                                                                    #TOTAL=0
                                                                    cv2.putText(frame, ""+ID_NAME, (10, 180),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                                                                else:
                                                                    cv2.putText(frame, "Face is not Real ", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                                                    cv2.putText(frame, "LAST ID VISITED: "+ID_NAME, (10, 160),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                           
                if len(rects) == 0:   #   number of blink remain zero if number of face are zero
                    TOTAL=0


                if z !=0:
                                                                        
                    cv2.putText(frame, "ATTENDANCE LOGGED: "+ID_NAME, (10, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                    z=z-1
                    #print(z)
                #z=0
                if z==0:
                    cv2.putText(frame, "", (10, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                if b !=0:
                    cv2.putText(frame, "ATTENDANCE ALREADY LOGGED IN "+ID_NAME, (10, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                    b=b-1
                if b==0:
                    cv2.putText(frame, "", (10, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                

                if len(rects)>1:
                    cv2.putText(frame, "Multiple face found : INVALID  ", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
              
                cv2.imshow("Frame",frame)
                
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                                break
                if key == ord("w"):
                    TOTAL=0
                if len(ID_NAME)==5:
                    break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()

