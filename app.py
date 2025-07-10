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

        print("📥 Received request")
        print(f"📂 img1: {img1.filename if img1 else 'None'}")
        print(f"📂 img2: {img2.filename if img2 else 'None'}")

        if not img1 or not img2:
            print("⚠️ Missing image(s)")
            return jsonify({"error": "img1 and img2 are required"}), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp1, \
             tempfile.NamedTemporaryFile(suffix=".jpg") as temp2:

            img1.save(temp1.name)
            img2.save(temp2.name)
            print("✅ Saved both images to temp files")

            result = DeepFace.verify(
                temp1.name,
                temp2.name,
                enforce_detection=False
            )

        print("✅ DeepFace comparison complete")
        return jsonify({
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"]
        })

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    # รันบนเครื่องเรา (development) — เปิดบนทุก interface ที่พอร์ต 5000
    app.run(host="0.0.0.0", port=5000, debug=True)