from deepface import DeepFace
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    img1_path = request.form['img1']
    img2_path = request.form['img2']
    result = DeepFace.verify(img1_path, img2_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)