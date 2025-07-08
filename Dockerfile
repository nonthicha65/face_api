FROM python:3.10-slim

# ✅ ติดตั้ง libgl1 เพิ่มเพื่อให้ cv2 ใช้งานได้
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD exec gunicorn app:app --bind 0.0.0.0:$PORT