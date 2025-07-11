from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile

app = Flask(__name__)

# ------------------------------------------------------------------
# Lazy-load resources  ➜  โหลดเมื่อมี request ครั้งแรกเท่านั้น
# ------------------------------------------------------------------
MODEL    = None      # จะกลายเป็นโมเดล SFace หลังโหลด
DETECTOR = None      # opencv face-detector (เบาและไม่ต้องใช้ GPU)

@app.route("/")
def home():
    return "API is live!"

@app.route("/verify", methods=["POST"])
def verify():
    global MODEL, DETECTOR
    try:
        # ------------------------------------------------------------------
        # 1) ตรวจสอบไฟล์รูป
        # ------------------------------------------------------------------
        img1, img2 = request.files.get("img1"), request.files.get("img2")
        if not img1 or not img2:
            return jsonify({"error": "img1 and img2 are required"}), 400

        # ------------------------------------------------------------------
        # 2) Lazy-load (โหลดโมเดล / detector ครั้งแรกเท่านั้น)
        # ------------------------------------------------------------------
        if MODEL is None:
            MODEL = DeepFace.build_model("SFace")       # ~25 MB  ใช้ RAM น้อยมาก
    
        # ------------------------------------------------------------------
        # 3) เซฟไฟล์ลง temp แล้วเรียก deepface.verify
        # ------------------------------------------------------------------
        with tempfile.NamedTemporaryFile(suffix=".jpg") as t1, \
             tempfile.NamedTemporaryFile(suffix=".jpg") as t2:
            img1.save(t1.name)
            img2.save(t2.name)

            result = DeepFace.verify(
                t1.name, t2.name,
                model=MODEL,
                detector_backend = "opencv", 
                enforce_detection=False
            )

        # ------------------------------------------------------------------
        # 4) ส่งผลลัพธ์กลับ
        # ------------------------------------------------------------------
        return jsonify({
            "verified":  result["verified"],
            "distance":  result["distance"],
            "threshold": result["threshold"]
        })

    except Exception as e:
        # log error ใน stdout ของ container จะเห็นใน Render logs
        print("❌ DeepFace error:", e)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# Local run (ใช้ gunicorn ใน Docker/prod อยู่แล้ว)
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)