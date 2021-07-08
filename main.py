import os
import cv2
import urllib
import numpy as np
from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen
from flask import Flask, render_template, Response, request, redirect, flash, url_for
from camera import VideoCamera
from Graphical_Visualisation import Emotion_Analysis


app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def gen(camera):

    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/')
def home():
    return render_template('Start.html')

@app.route('/')
def Start():
    return render_template('Start.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/RealTime', methods=['POST'])
def RealTime():
    return render_template('RealTime.html')


@app.route('/takeimage', methods=['POST'])
def takeimage():
 
    v = VideoCamera()
    _, frame = v.video.read()
    save_to = "static/"
    cv2.imwrite(save_to + "capture" + ".jpg", frame)

    result = Emotion_Analysis("capture.jpg")


    if len(result) == 1:
        return render_template('NoDetection.html', orig=result[0])

    # sentence = mood(result[3])
    # activity = activities(result[3])
    # link = provide_url(result[3])
    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2])

@app.route('/ManualUpload', methods=['POST'])
def ManualUpload():
     return render_template('ManualUpload.html')


@app.route('/uploadimage', methods=['POST'])
def uploadimage():
    

    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

  
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result = Emotion_Analysis(filename)
            
            if len(result) == 1:

                return render_template('NoDetection.html', orig=result[0])

 
            return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2])

if __name__ == '__main__':
    app.run(debug=True)
