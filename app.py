import os
# ---------------- TensorFlow/DeepFace – force CPU ----------------
# ตั้งค่า env BEFORE import tensorflow / deepface
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ปิดการมองเห็น GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"         # ลด log verbosity ของ TF

from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile

app = Flask(__name__)

# ========== โหลดโมเดลและตัวตรวจครั้งเดียวตอนบูต ==========
MODEL    = DeepFace.build_model("ArcFace")        # เบากว่า Facenet+RetinaFace
DETECTOR = DeepFace.build_detector("opencv")      # ไม่ต้องใช้ RetinaFace

@app.route("/")
def home():
    return "API is live!"

@app.route("/verify", methods=["POST"])
def verify():
    try:
        img1, img2 = request.files.get("img1"), request.files.get("img2")
        if not img1 or not img2:
            return jsonify({"error": "img1 and img2 are required"}), 400

        # --------- เซฟรูปลงไฟล์ชั่วคราว ----------
        with tempfile.NamedTemporaryFile(suffix=".jpg") as t1, \
             tempfile.NamedTemporaryFile(suffix=".jpg") as t2:

            img1.save(t1.name)
            img2.save(t2.name)

            # --------- เปรียบเทียบใบหน้า ----------
            result = DeepFace.verify(
                t1.name, t2.name,
                model=MODEL,
                detector_backend=DETECTOR,
                enforce_detection=False
            )

        return jsonify({
            "verified":  result["verified"],
            "distance":  result["distance"],
            "threshold": result["threshold"]
        })

    except Exception as e:
        # log ลง stdout เพื่อให้ดูใน render log ได้
        print("❌ Verification error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------- Local / Render run -----------------------
if __name__ == "__main__":
    # ถ้า Render รันจะมีตัวแปร PORT ใส่มาให้
    port = int(os.environ.get("PORT", 5000))
    # debug=False เพื่อให้ gunicorn จับเวลา worker ได้ถูก
    app.run(host="0.0.0.0", port=port, debug=False)