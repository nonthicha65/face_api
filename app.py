from deepface import DeepFace
from flask import Flask, request, jsonify
import tempfile

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    # รับไฟล์ภาพจาก form-data
    img1_file = request.files.get("img1")
    img2_file = request.files.get("img2")

    # สร้างไฟล์ชั่วคราวเพื่อให้ DeepFace ใช้งาน
    with tempfile.NamedTemporaryFile(suffix=".jpg") as img1_temp, \
         tempfile.NamedTemporaryFile(suffix=".jpg") as img2_temp:

        img1_file.save(img1_temp.name)
        img2_file.save(img2_temp.name)

        result = DeepFace.verify(img1_temp.name, img2_temp.name)

    # คืนผลลัพธ์ JSON กลับ
    return jsonify({
        "verified": result["verified"],
        "distance": result["distance"],
        "threshold": result["threshold"]
    })

@app.route('/')
def health():
    return 'API is live!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
