import io
from PIL import Image
import datetime
import cv2
import os

import torch
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)

# Function to Load YOLO model
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="D:/TugasKampus/TA/skin_problems_detect/model/best.pt", force_reload=True)
    model.eval()
    return model

# Set video input to webcam
video = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier()

# Run the load_model function
model = load_model()

# Save current date time
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

# Default route
@app.route("/", methods=["GET", "POST"])

# Function to predict using image
def predict():
    if request.method == "POST":

        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        # If the uploaded content is an image
        if file.filename.split('.').pop().lower() in ['jpg', 'jpeg']:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            # results = model([img])
            results = model([img], size=1920)
            results.render()  # updates results.img with boxes and labels
            now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
            img_savename = f"static/{now_time}.png"
            Image.fromarray(results.ims[0]).save(img_savename)
            return redirect(img_savename)
        
        # If the uploaded content is a video
        elif file.filename.split('.').pop().lower() in ['mp4']:
            video_filename = f"static/{file.filename}"
            file.save(video_filename)

            base_dir = 'D:\TugasKampus\TA\skin_problems_detect'
            full_dir = os.path.join(base_dir, video_filename)

            video = cv2.VideoCapture(full_dir)
            
            return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template("index.html")

# Function to predict using video
def gen(video):
    while True:
        success, image = video.read()
        frame_gray=image

        # YOLO
        faces = model([frame_gray], size=1920)
        faces.render()

        print(faces)

        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for webcam detection
@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run() 