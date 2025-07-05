from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import tempfile
import requests

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status": "Face API is running"})

@app.route("/verify-face", methods=["POST"])
def verify_face():
    try:
        data = request.get_json()
        image1_url = data["image1_url"]
        image2_url = data["image2_url"]

        def download_image(url):
            response = requests.get(url)
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp.write(response.content)
            temp.close()
            return temp.name

        img1 = download_image(image1_url)
        img2 = download_image(image2_url)

        result = DeepFace.verify(img1, img2, enforce_detection=False)

        return jsonify({
            "match": result["verified"],
            "distance": result["distance"],
            "message": "Face verification completed"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
