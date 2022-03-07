import cv2
from face_recognition.api import face_encodings  #To Read The Image We can Use CV2 Modules
import numpy as np
import face_recognition
import os #read face modules
from datetime import datetime

# Image read out
path = 'images'
images = []  #Create a list to store images
personName = [] #To store Name of Person
mylist = os.listdir(path)
print(mylist) # It return the list of Name Present in Image File

#Seprate Name from Images File
for cu_img in mylist:
    current_img=cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0]) #[0] it is First name of images
print(personName) # It Return The Name Of Person That is Present in our Image File

#Face incoding :- It means that it can create one image into 120 unique point of  images to read images easily
def faceEncodings(images):
    encodeList = []
    for img in images:    #IMAGES  are in bjr format due to cv2 retyrn in bjr format 
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # img can be converted into RGB format
          encode = face_recognition.face_encodings(img)[0]
          encodeList.append(encode)
    return encodeList   # Widthout you cann't read  image distance or face comparision

encodeListKnown=faceEncodings(images)   #It Uses the HOG Algorithm
print("All Encoding Completed")

#This code is use to mark attendence  in the file
def attendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList=[]
        for line in myDataList:
            entry= line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            time_now = datetime.now()
            tstr = time_now.strftime('%H:%M:%S')
            dstr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tstr},{dstr}')
            f.writelines(f'\n')
        
        elif name in nameList:
            f.readlines()


#Camera read
cap = cv2.VideoCapture(0)  #id of camera in laptop is 0, enternal webcame id 1
while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25,0.25)
    faces = cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)

    facesCurrentFrame  = face_recognition.face_locations(faces) # it can decet face in current location
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)
    #compare & Distance Find Out
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # face  distance between two face is minimum it can read but distance is maximum it cannot read

        matchIndex = np.argmin(faceDis) # it can return the index value
        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            #print(name) # It print the Name of person that is capture in the camera if its photo is avillable in the image file
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )
            attendance(name)

    cv2.imshow("Camera",frame)
    # This code is use to stop the program to running using enter key
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()   



    







input()

