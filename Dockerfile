# ใช้ Python 3.10 แบบ slim เพื่อลดขนาด
FROM python:3.10-slim

# ติดตั้ง dependency สำหรับ DeepFace และ TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ตั้งค่า working directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดไปยัง container
COPY . .

# อัปเกรด pip และติดตั้ง dependencies จาก requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# รันแอปด้วย gunicorn ที่พอร์ต 10000 (Render ใช้ env PORT)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]