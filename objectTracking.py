from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__, template_folder='demo')

# Folder to save uploaded videos
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the YOLO model
model=YOLO(os.path.join('0bb.pt'))

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video upload
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return render_template('index.html')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('video_feed', filename=file.filename))

# Function to generate frames from video
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the YOLO model on the frame
        results = model.track(frame_rgb, persist=True)

        # Extract the processed frame with bounding boxes (visualization)
        processed_frame = results[0].plot()

        # Convert back to BGR for OpenCV display
        #processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format to be streamed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to stream the video
@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
