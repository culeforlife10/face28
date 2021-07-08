import cv2
import numpy as np

from model import FacialExpressionModel


model = FacialExpressionModel("model.json", "model_weights.h5")

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):


        
        _, frame = self.video.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scaleFactor = 1.3
        minNeighbors = 5
        faces = facec.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

        for (x, y, w, h) in faces:


            roi = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            prediction = model.predict_emotion(
                roi[np.newaxis, :, :, np.newaxis])
            Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!",
                       "Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}
            Text = str(prediction) + Symbols[str(prediction)]
            Text_Color = (180, 105, 255)

            Thickness = 4
            Font_Scale = 2
            Font_Type = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame, Text, (x, y), Font_Type,
                        Font_Scale, Text_Color, Thickness)
            xc = int((x + x+w)/2)
            yc = int((y + y+h)/2)
            radius = int(w/2)

            cv2.circle(frame, (xc, yc), radius, (0, 255, 0), Thickness)
        _, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()
