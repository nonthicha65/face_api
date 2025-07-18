FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# สำคัญ!  อย่าฮาร์ดโค้ด 10000
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120