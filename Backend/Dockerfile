# Use Python 3.12 with TensorFlow/Keras versions compatible with this model format
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/model/model.h5 \
    PORT=5000

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Start without Flask debug/reloader
CMD ["python", "-c", "import os, app; app.app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=False)"]
