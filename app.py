import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import time

# Defining Flask App
app = Flask(__name__)

nimgs = 10

# Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv', 'w') as f:
        f.write('Name,Roll,Time,Report')


# total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    # iris = datasets.load_iris()
    # x, y = iris.data[:, :], iris.target
    # faces, xfaces, labels, y_labels = train_test_split(x, y, stratify=y, random_state=0, train_size=0.7)
    #
    # scaler = preprocessing.StandardScaler().fit(faces)
    # faces = scaler.transform(faces)
    # xfaces = scaler.transform(xfaces)
    print("accuracy_score")

    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    y_pred = knn.predict(faces)
    # acc_knn = accuracy_score(labels, knn_model.predict(faces))
    print(accuracy_score(labels, y_pred))
    joblib.dump(knn, 'static/face_recognition_model.pkl')

    # print('KNN training accuracy = ' + str(100 * acc_knn) + '%')
    # print(accuracy_score(labels, y_pred))

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    reports = df['Report']
    l = len(df)
    return names, rolls, times, reports, l


# Add specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    report = 'Menggunakan Lab'

    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{report}')


# Selesai specific user
def finish_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    report = 'Selesai'

    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if int(userid) in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{report}')


# ROUTING FUNCTIONS#

# Main page
@app.route('/home')
def home():
    names, rolls, times, reports, l = extract_attendance()
    return render_template('front.html', names=names, rolls=rolls, times=times, reports=reports, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2())

# Front page
@app.route('/')
def front():
    names, rolls, times, reports, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, reports=reports, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2())

# Run Menyalakan Kamera Button
@app.route('/start', methods=['GET'])
def start():
    close = False
    #print("accuracy_score")
    timeout = 0
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('front.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if not close:
            if extract_faces(frame) != ():
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
                close = True
                timeout = time.time() + 5
                # time.sleep(5)
                # break
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break
        elif time.time() > timeout:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, reports, l = extract_attendance()
    return render_template('front.html', names=names, rolls=rolls, times=times, reports=reports, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2())


# Run Selesai Button
@app.route('/done', methods=['GET'])
def done():
    close = False
    timeout = 0
    print("run")
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print("done")
        return render_template('front.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')
    print("here")
    #
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if not close: #detect
            if extract_faces(frame) != ():
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                finish_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
                close = True
                timeout = time.time() + 5
                # time.sleep(5)
                # break
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                print("here")
                break
        elif time.time() > timeout: # cam close after 5 secs
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, reports, l = extract_attendance()
    return render_template('front.html', names=names, rolls=rolls, times=times, reports=reports, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2())


# Run add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, reports, l = extract_attendance()
    return render_template('front.html', names=names, rolls=rolls, times=times, reports=reports, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


if __name__ == '__main__':
    app.run(debug=True)