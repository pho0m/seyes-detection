import torch
import numpy as np
import base64

from io import BytesIO
from flask import Flask , json, request ,send_file
from werkzeug.exceptions import HTTPException
from PIL import Image

app = Flask(__name__)

@app.route('/')
def healthCheck():  
   return 'Hi from seyes detect micro service' 

@app.route('/detect', methods=['POST'])
def detectImage():  
   ID = request.form.get("id")
   image = request.files['image']
  #  nparr = np.fromstring(image, np.uint8)
   print(ID)
   print(image)
  #  print(nparr)
   img = Image.open(image)  # load with Pillow

   model = torch.hub.load('ultralytics/yolov5', 'custom', 'bestV5.pt')

   # Images
   imgs = [img]   
   # Inference
   results = model(imgs)

   # Results
   results.print()
   # results.show()  # or .show()
   
   results.ims # array of original images (as np array) passed to model for inference
   results.render()  # updates results.ims with boxes and labels
   
   b64Image = ""
   
   for im in results.ims:
       buffered = BytesIO()
       im_base64 = Image.fromarray(im)
       im_base64.save(buffered, format="JPEG")
       b64Image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
 
   uri = ("data:image/jpeg" +";" +
       "base64," + b64Image)
    
   return uri

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5000, debug=True)

