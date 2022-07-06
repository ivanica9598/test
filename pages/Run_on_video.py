import face_recognition
import cv2
import numpy as np
import os
import datetime
import streamlit as st

@st.cache
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in line:
            now = datetime.datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')

path = 'images'
images = []  # LIST CONTAINING ALL THE IMAGES
classNames = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS NAMES
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))

for x, cl in enumerate(myList):
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images)
print('Encodings Complete')

stframe = st.empty()
start_test = st.sidebar.button('Start test')
end_test = st.sidebar.button('End test')

if not "test_active" in st.session_state:
    st.session_state["test_active"] = False

if start_test :
    st.session_state["test_active"] = True

if end_test:
    st.session_state["test_active"] = False

if start_test:
    factor = 4
    cap = cv2.VideoCapture(0)
    while st.session_state["test_active"]:
        success, img = cap.read()
        imgFrame = cv2.resize(img, (0, 0), fx=1/factor, fy=1/factor)
        imgFrame = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2RGB)

        facesFrame = face_recognition.face_locations(imgFrame)
        encodesFrame = face_recognition.face_encodings(imgFrame, facesFrame)

        for encodeFace, faceLoc in zip(encodesFrame, facesFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)
            if faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                name = 'Unknown'

            y1, x2, y2, x1 = faceLoc
            # top, right, bottom, left
            y1, x2, y2, x1 = y1 * factor, x2 * factor, y2 * factor, x1 * factor
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

            #cv2.imshow('Webcam', img)
            stframe.image(img)

        if end_test:
            st.session_state["test_active"] = False
            break
    cap.release()
    cv2.destroyAllWindows()



