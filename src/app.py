import os
from flask import Flask, request, redirect, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS

from mock_data import model
from file_util import allowed_file
from interaction_model_generator import InteractionModelGenerator
from paraphraser.para import Paraphraser

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'download'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


paraphraser = Paraphraser()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/v1/interaction-model', methods=['GET'])
def api_all():
    resp = jsonify({'data': model})
    return resp


@app.route('/api/v1/file-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        extractor = InteractionModelGenerator(UPLOAD_FOLDER, filename, paraphraser)
        interaction_model = extractor.generate()
        resp = jsonify({'message': 'File successfully uploaded', 'data': interaction_model})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp


@app.route('/api/v1/file-download', methods=['GET'])
def download_file():
    return send_file('../download/convo.zip', attachment_filename='convo.zip')

if __name__ == '__main__':
    app.run()
