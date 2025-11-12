import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# We must import from the 'src' folder
from src.gesture_module import GestureModule

# Initialize our Flask app
app = Flask(__name__)
# Allow our front-end to talk to this server
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "ok", "endpoints": ["/recognize"]}), 200

# Load our gesture module "brain" one time
print("Loading gesture module...")
# This uses your modified gesture_module.py
gesture_recognizer = GestureModule() 
print("Module loaded. Server is ready.")


@app.route('/recognize', methods=['POST'])
def recognize_gesture():
    """This is the API endpoint our front-end will call."""
    
    # Get the image data from the web request
    data = request.json
    # The image comes as a "base64" string, we need to decode it
    img_data = base64.b64decode(data['image'].split(',')[1])
    
    # Convert the image data to a numpy array for OpenCV/MediaPipe
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the frame using our modified module
    # We must flip it because webcam images are mirrored
    frame_flipped = cv2.flip(frame, 1)
    
    # Get the result dictionary
    result = gesture_recognizer.process_frame(frame_flipped)
    
    # Send the result back to the front-end as JSON
    return jsonify(result)

if __name__ == '__main__':
    # Run the server on port 5000 without the auto-reloader
    app.run(debug=True, port=5000, use_reloader=False)