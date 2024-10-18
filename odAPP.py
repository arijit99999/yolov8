import cv2 as cv
from ultralytics import YOLO
import io,base64
from PIL import Image
from flask import render_template,request,Flask
import os

app=Flask(__name__,template_folder='demo/')
model=YOLO('best.pt')
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route ('/')
def home():
    return render_template('test.html')
@app.route('/pred', methods=['POST'])
def marks():
    file=request.files['image']
    file.save(os.path.join('uploads',file.filename))
    file_path=os.path.join('uploads',file.filename)
    image=cv.imread(file_path) #load image
    result=model.predict(image,conf=.5)
    os.makedirs('predImage/', exist_ok=True)
    for i in result:
        i.save(os.path.join('predImage',file.filename))
    img=cv.imread(os.path.join('predImage',file.filename))
    segmented_image=Image.fromarray(img[:,:,::-1])
    buffered = io.BytesIO()
    segmented_image.save(buffered, format="JPEG")
    buffered.seek(0)
    img_str = base64.b64encode(buffered.getvalue()).decode() #encoding 
    img2=cv.imread(file_path)
    image2=Image.fromarray(img2[:,:,::-1])
    buffered = io.BytesIO() #create object
    image2.save(buffered, format="JPEG") #save in bytes    
    plot_url1= base64.b64encode(buffered.getvalue()).decode()
    return render_template('res.html',x=img_str,y=plot_url1)

if __name__=='__main__':
    app.run()