from flask import Flask, render_template, redirect, request, jsonify, send_file
import os
import face2fate
import facerating
import uuid

UPLOAD_FOLDER = './faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def saveFile(file_name):
	file = request.files[file_name]
	# if user does not select file, browser also
	# submit an empty part without filename
	if file.filename == '':
		return redirect(request.url)
	if file and allowed_file(file.filename):
		face_path = app.config['UPLOAD_FOLDER']
		if not os.path.isdir(face_path):
			os.mkdir(face_path)
		file_path = os.path.join(face_path, file_name + str(uuid.uuid4()) + ".jpg")
		file.save(file_path)
		return file_path


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/get_crop')
def get_image():
	filename = "crop_" + request.args.get('image_name')
	return send_file("faces/" + filename, mimetype='image/jpeg')


@app.route('/detect', methods=['POST'])
def detect():
	if 'face' not in request.files:
		return jsonify(message="Invalid Params")

	face_path = saveFile('face')
	face_desc = face2fate.fortuneTelling(face_path)
	print("face_desc", face_desc)
	if face_desc == 0:
		return jsonify(message="Not detect face")
	elif face_desc == 1:
		return jsonify(message="Many face")
	else:
		face_score = facerating.predict(face_path)
		print("face_score", face_score)
		return jsonify(message="Success",face_desc=face_desc,face_score=face_score)

if __name__ == '__main__':
	app.run(threaded=True, port=5000)
