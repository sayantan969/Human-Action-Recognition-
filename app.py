from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from model_utils import predict_action_from_video
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
@app.route("/predict", methods=["POST"])

def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file part"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    label, confidence = predict_action_from_video(video_path)

    confidence = float(confidence)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)