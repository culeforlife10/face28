import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FacialExpressionModel
test_model = FacialExpressionModel("model.json", "model_weights.h5")

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def Emotion_Analysis(img):
    path = "static/" + str(img)
    image = cv2.imread(path)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.3
    minNeighbors = 5
    faces = facec.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

    if len(faces) == 0:
        return [img]

    for (x, y, w, h) in faces:
        roi = gray_frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        prediction = test_model.predict_emotion(
            roi[np.newaxis, :, :, np.newaxis])
        Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!",
                   "Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}
    
        Text = str(prediction) + Symbols[str(prediction)]
        Text_Color = (180, 105, 255)

        Thickness = 4
        Font_Scale = 2
        Font_Type = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(image, Text, (x, y), Font_Type,
                    Font_Scale, Text_Color, Thickness)
        xc = int((x + x+w)/2)
        yc = int((y + y+h)/2)
        radius = int(w/2)

        cv2.circle(image, (xc, yc), radius, (0, 255, 0), Thickness)

        path = "static/" + "pred" + str(img)
        cv2.imwrite(path, image)

        EMOTIONS = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

        preds = test_model.return_probabs(roi[np.newaxis, :, :, np.newaxis])
        data = preds.tolist()[0]

        fig = plt.figure(figsize=(8, 5))
        plt.bar(EMOTIONS, data, color='green',
                width=0.4)
        plt.xlabel("Types of Emotions")
        plt.ylabel("Probability")
        plt.title("Facial Emotion Recognition")
        path = "static/" + "bar_plot" + str(img)
        plt.savefig(path)

    return ([img, "pred" + img, "bar_plot" + img, prediction])
