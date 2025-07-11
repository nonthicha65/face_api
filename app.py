import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to run on CPU only

from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile

app = Flask(__name__)

# === Load once on boot ===
MODEL = DeepFace.build_model("ArcFace")        # lighter than Facenet
DETECTOR = DeepFace.build_detector("opencv")   # avoid GPU-dependent detector

@app.route("/")
def home():
    return "API is live!"

@app.route("/verify", methods=["POST"])
def verify():
    try:
        img1 = request.files.get("img1")
        img2 = request.files.get("img2")
        if not img1 or not img2:
            return jsonify({"error": "img1 and img2 are required"}), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg") as t1, \
             tempfile.NamedTemporaryFile(suffix=".jpg") as t2:
            img1.save(t1.name)
            img2.save(t2.name)

            result = DeepFace.verify(
                t1.name,
                t2.name,
                model=MODEL,
                detector_backend=DETECTOR,
                enforce_detection=False,
            )

        return jsonify(
            {
                "verified": result["verified"],
                "distance": result["distance"],
                "threshold": result["threshold"],
            }
        )
    except Exception as e:
        print("❌", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)