
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes):
    # features is a 1D array, reshape so we have a matrix
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    graphs = []

    # split out the data from all axes
    for ax in range(0, len(axes)):
        X = []
        for ix in range(0, raw_data.shape[0]):
            X.append(float(raw_data[ix][ax]))

        # X now contains only the current axis
        fx = np.array(X)

        # process the signal here
        fx = fx * scale_axes

        # we need to return a 1D array again, so flatten here again
        for f in fx:
            features.append(f)

    return {
        'features': features,
        'graphs': graphs,
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            'type': 'flat',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                'width': len(features)
            }
        }
    }

def test():

    cap = cv2.VideoCapture("cvtest.mp4")
    print(cap.isOpened())   # True = read video successfully. False - fail to read video.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output_video.avi", fourcc, 20.0, (640, 360))
    print(out.isOpened())  # True = write out video successfully. False - fail to write out video.
    cap.release()
    out.release()


def findface():
    face_cascade = cv2.CascadeClassifier(
        'C:/Users/Ken/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/Ken/Downloads/opencv/sources/data/haarcascades/haarcascade_eye.xml')
    cap = cv2.VideoCapture("kentest3.mp4")
#    cap = cv2.VideoCapture(0)
    imgctr = 0
    id = 0
    while 1:
        id += 1
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            cv2.imwrite(
                str(id) + ".jpg",
                roi_color)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                imgctr += 1
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.imwrite(
                    str(id) + "." + str(imgctr) + ".jpg",
                    roi_color)

        if str(len(faces) > 0):
            print("found " + str(len(faces)) + " face(s)")
            cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def findfaceinpic():
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    img = cv2.imread('test.jpg')
    print('img as')
    print(type(img))
    print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('gray as')
    print(type(gray))
    print(gray)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
  #  cv2.imshow('img', img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # print("found " + str(len(faces)) + " face(s)")
       # cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    status = cv2.imwrite('faces_detected1.jpg', img)
    print('flattened :')
    a = np.array(roi_gray)
    a = a.flatten()
    print(a)
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    findfaceinpic()
    # test()
    # generate_features()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
