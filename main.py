import torch
import base64
import os
import io
import matplotlib.pyplot as plt

from io import BytesIO
from flask import Flask , json, request
from werkzeug.exceptions import HTTPException
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv 

load_dotenv()

UPLOAD_FOLDER = '/Users/bubpa/workspace/seyes-detection/model'
ALLOWED_EXTENSIONS = set(['pt'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1) [1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def healthCheck():  
   return 'Hi from seyes detect micro service'
 

@app.route('/detect', methods=['POST'])
def detectImage():  
   image = request.files['image']
   print(image)
   img = Image.open(image)
   
   model = torch.hub.load('ultralytics/yolov5', 'custom', 'model/best.pt')


   imgs = [img]   
   results = model(imgs)
   results.ims
   results.render()
   output_html = results.render()
   output_html = output_html.replace("<style>", "<style> .d3graph rect {stroke: red !important;} ")
   

   df = results.pandas().xyxy[0]
   
   isDetec = True
   
   if df.empty:
    isDetec = False
    person_count = 0
    com_on_count = 0
    acc = 0
    status_detec = "UnDetected"   
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
   else:
    isDetec = True
    person_count = str(results.pandas().xyxy[0].value_counts('name').person)
    com_on_count = str(results.pandas().xyxy[0].value_counts('name').com_on)
    acc = (sum(results.pandas().xyxy[0].value_counts('confidence').index)/sum(results.pandas().xyxy[0].value_counts('confidence')))*100
    status_detec = "Detected"
    
   
   now = datetime.now()
   date = now.strftime("%d/%m/%Y")
   time = now.strftime("%H:%M:%S")
   
   b64Image = ""
   
   for im in results.ims:
       buffered = BytesIO()
       im_base64 = Image.fromarray(im)
       im_base64.save(buffered, format="JPEG")
       b64Image = base64.b64encode(buffered.getvalue()).decode('utf-8')
       
   url = "" 
   if isDetec :
    url = ("data:image/jpeg" +";" +
       "base64," + b64Image)
   else :
    url = ("data:image/jpeg" +";" +
       "base64," + base64_data)
    

   res = {
    #    "phote_url":url,
           "person_count" : person_count,
           "com_on_count" : com_on_count,
           "accuracy" : '{0:.4g}'.format(acc),
           "date" : date,
           "time" : time,
           "status_detec" : status_detec}
    
   print(res)
   return  url


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    response = e.get_response()
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.route('/upload', methods=['POST'])
def upload_media():
    if 'file' not in request.files: 
        return jsonify({'error': 'media not provided'}), 400
    file= request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    if file and allowed_file(file.filename):
        filename = "best.pt"
        file.save(os.path.join(app.config ['UPLOAD_FOLDER'], filename)) 
    return jsonify({'msg': 'uploaded successfully'})

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=os.getenv("PORT_DETEC") ,debug=True)