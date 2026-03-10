## Deploy Backend On EC2 With Docker

### 1) Connect and prepare
```bash
ssh -i <key.pem> ubuntu@<EC2_PUBLIC_IP>
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

### 2) Clone project and build image
```bash
cd ~
git clone <your-repo-url>
cd EnergySaver_Frontend/Backend
docker build --platform linux/amd64 -t energy-backend:latest .
```

### 3) Run container
```bash
docker run -d \
  --name energy-backend \
  --restart unless-stopped \
  -p 5000:5000 \
  energy-backend:latest
```

### 4) Verify model and API
```bash
curl http://127.0.0.1:5000/status
```
Expected:
- `"model_loaded": true`
- `"model_path": "/app/model/model.h5"` (or another resolved path)
- `"model_error": null`

### 5) Open EC2 security group
Allow inbound TCP `5000` from your client IP (or load balancer).

## Why errors happened

- Library version/build errors:
  - The saved model was created with newer Keras metadata (includes `optional` and `batch_shape` fields).
  - Old runtime (`tensorflow==2.15.0`) cannot deserialize that format.
  - This project now uses `tensorflow==2.20.0` on Python `3.12`.
  - On many EC2 ARM instances (`t4g`, `m7g`) you can hit wheel incompatibility.

- Model not loading:
  - Code originally only looked at `/models/model.h5` (container mount path).
  - The model file in repo is `model/model.h5`, so local/container runs without that mount failed.
  - The app now resolves model path via:
    1. `MODEL_PATH` env var
    2. `/models/model.h5`
    3. `/app/model/model.h5` (repo path inside container)
