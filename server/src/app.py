from .utils import DatasetHandler, CNN, ModelBuilder, DEVICE, torch
from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
from PIL import Image

import os

root = os.path.dirname(os.path.dirname(__file__))
builder = ModelBuilder(CNN().to(DEVICE))
builder.load(path=os.path.join(root, 'model.pt'))

routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    return jsonify({'status_code': 200})

@routes.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        label, probabilities = builder.predict(file)
        print(f'Prediction: {label}\nProbabilities: {probabilities}')
        return jsonify({'label': label, 'probabilities': probabilities}), 200
    except Exception as e:
        return jsonify({'error': e}), 500

def serve():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(routes)
    return app
