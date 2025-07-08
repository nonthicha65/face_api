from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile

app = Flask(__name__)

@app.route('/')
def home():
    return "API is live!"

@app.route('/verify', methods=['POST'])
def verify():
    try:
        img1 = request.files.get("img1")
        img2 = request.files.get("img2")

        if not img1 or not img2:
            return jsonify({"error": "img1 and img2 are required"}), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp1, \
             tempfile.NamedTemporaryFile(suffix=".jpg") as temp2:
            img1.save(temp1.name)
            img2.save(temp2.name)

            print("🧠 Comparing faces...")
            result = DeepFace.verify(temp1.name, temp2.name, enforce_detection=False)

        return jsonify({
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"]
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500