from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile

app = Flask(__name__)

@app.route("/")
def home():
    return "API is live!"

@app.route("/verify", methods=["POST"])
def verify():
    img1 = request.files.get("img1")
    img2 = request.files.get("img2")

    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp1, \
         tempfile.NamedTemporaryFile(suffix=".jpg") as temp2:
        img1.save(temp1.name)
        img2.save(temp2.name)
        result = DeepFace.verify(temp1.name, temp2.name)

    return jsonify({
        "verified": result["verified"],
        "distance": result["distance"],
        "threshold": result["threshold"]
    })

if __name__ == "__main__":
    app.run(debug=True)