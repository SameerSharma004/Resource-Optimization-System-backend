from flask import Flask, jsonify, request, send_from_directory, abort
import os
import numpy as np
import datetime
import jwt
from functools import wraps
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import joblib

def load_local_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"')

load_local_env()

try:
    import psutil
except ImportError:
    psutil = None

app = Flask(__name__)
CORS(app)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-12345")
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://sam18112k4_db_user:lfZ3QKvlgznCpmTU@rosuserdata.pio2vka.mongodb.net/?appName=rosuserdata")
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "downloads")
users_collection = None

print(f"DEBUG: Attempting to connect to MongoDB URI: {MONGO_URI[:30]}...")

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    # The ismaster command is cheap and does not require auth.
    mongo_client.admin.command('ismaster')
    db = mongo_client["ros"]
    users_collection = db["userdata"]
    users_collection.create_index("email", unique=True)
    print("SUCCESS: MongoDB connected successfully")
except Exception as e:
    print(f"ERROR: MongoDB connection failed: {e}")
    users_collection = None
DEFAULT_MODEL_CANDIDATES = [
    "/models/model.h5",
    os.path.join(os.path.dirname(__file__), "model", "model.h5"),
]
SEQUENCE_LENGTH = 10
MODEL_LOADED = False
MODEL_LOAD_ERROR = None
MODEL_PATH = None
try:
    env_model_path = os.getenv("MODEL_PATH")
    candidate_paths = [env_model_path] if env_model_path else DEFAULT_MODEL_CANDIDATES
    MODEL_PATH = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if not MODEL_PATH:
        raise FileNotFoundError("Model file not found")
    model = load_model(MODEL_PATH, compile=False)
    MODEL_LOADED = True
    print("AI Model loaded successfully")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print("Model load failed:", e)
    model = None
try:
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.save")
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully from:", scaler_path)
except Exception as e:
    print("Scaler load failed:", e)
    scaler = None
USER_SESSIONS = {}
def get_user_session(email):
    if email not in USER_SESSIONS:
        USER_SESSIONS[email] = {
            "LAST_SYSTEM_DATA": None,
            "SYSTEM_HISTORY": [],
            "LAST_PREDICTION": None,
            "FEATURE_BUFFER": []
        }
    return USER_SESSIONS[email]

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"message": "Token missing"}), 401
        try:
            token = auth_header.split(" ")[1]
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = decoded
        except Exception:
            return jsonify({"message": "Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(DOWNLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

@app.route("/signup", methods=["POST"])
def signup():
    if users_collection is None:
        return jsonify({"error": "Database not connected"}), 500
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    if not name or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 409
    hashed_password = generate_password_hash(password)
    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    })
    return jsonify({"status": "registered"})

@app.route("/login", methods=["POST"])
def login():
    if users_collection is None:
        return jsonify({"error": "Database not connected"}), 500
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"message": "Missing email or password"}), 400
    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid login"}), 401
    token = jwt.encode({
        "user_id": str(user["_id"]),
        "email": user["email"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")
    return jsonify({
        "token": token,
        "user": {
            "name": user.get("name", "User"),
            "email": user.get("email")
        }
    })




@app.route("/analyze", methods=["POST"])

@token_required
def analyze():
    user_email = request.user["email"]
    print(f"DEBUG: Received /analyze request from {user_email}")
    session = get_user_session(user_email)

    if not MODEL_LOADED or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Store system data
    session["LAST_SYSTEM_DATA"] = data
    session["SYSTEM_HISTORY"].append({
        "time": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
        "cpu": data.get("cpu_usage", 0),
        "ram": data.get("memory_usage", 0),
        "net_up": data.get("net_upload_mbps", 0),
        "net_down": data.get("net_download_mbps", 0)
    })

    if len(session["SYSTEM_HISTORY"]) > 30:
        session["SYSTEM_HISTORY"].pop(0)

    try:
        # =========================
        # PREPARE FEATURES
        # =========================
        cpu_val = float(data.get("cpu_usage", 0))
        mem_val = float(data.get("memory_usage", 0))

        raw_features = [
            cpu_val,
            mem_val,
            float(data.get("net_upload_mbps", 0)),
            float(data.get("net_download_mbps", 0)),
            float(data.get("disk_read_mbps", 0)),
            float(data.get("disk_write_mbps", 0))
        ]

        # =========================
        # BUFFER (SEQUENCE)
        # =========================
        session["FEATURE_BUFFER"].append(raw_features)

        if len(session["FEATURE_BUFFER"]) > SEQUENCE_LENGTH:
            session["FEATURE_BUFFER"].pop(0)

        if len(session["FEATURE_BUFFER"]) < SEQUENCE_LENGTH:
            padded = [session["FEATURE_BUFFER"][0]] * (SEQUENCE_LENGTH - len(session["FEATURE_BUFFER"])) + session["FEATURE_BUFFER"]
        else:
            padded = session["FEATURE_BUFFER"]

        # =========================
        # SCALE + PREDICT
        # =========================
        future_cpu = cpu_val # Fallback
        if scaler and model:
            scaled = scaler.transform(padded)
            sequence = np.array(scaled).reshape(1, SEQUENCE_LENGTH, len(raw_features))
            prediction = model.predict(sequence, verbose=0)
            future_cpu = float(prediction[0][0]) * 100  # convert back
        

        # =========================
        # GEMINI-STYLE INTELLIGENCE
        # =========================
        top_proc_name = "N/A"
        top_proc_cpu = 0
        if psutil:
            try:
                processes = list(psutil.process_iter(['name', 'cpu_percent']))
                if processes:
                    top_p = max(processes, key=lambda x: x.info['cpu_percent'])
                    top_proc_name = top_p.info['name']
                    top_proc_cpu = top_p.info['cpu_percent']
            except: pass

        # Generate Narrative (The Gemini Touch)
        intelligence_report = []
        
        # 1. Executive Summary
        if future_cpu > 80:
            state = "Critical Saturation Imminent"
            summary = f"🚨 CRITICAL ALERT: Our neural engine predicts a severe resource spike. Total CPU utilization is projected to hit {future_cpu:.1f}% within the next observation window."
        elif future_cpu > 50:
            state = "High Load Warning"
            summary = f"⚠️ SYSTEM WARNING: Sustained high-load patterns detected. Expect resource pressure to stabilize around {future_cpu:.1f}%."
        elif future_cpu > 20:
            state = "Active Optimization"
            summary = f"✅ STABLE: The system is operating within balanced parameters. Projected overhead is nominal at {future_cpu:.1f}%."
        else:
            state = "Idle / Cooling"
            summary = f"❄️ OPTIMAL: System is currently in a low-power state. Neural forecast indicates minimal activity ({future_cpu:.1f}%)."

        intelligence_report.append(summary)

        # 2. Detailed Diagnostic
        diagnostic = f"DIAGNOSTIC: Current CPU is {cpu_val}% with {mem_val}% Memory saturation. "
        if top_proc_cpu > 10:
            diagnostic += f"The primary driver appears to be '{top_proc_name}' consuming {top_proc_cpu}% of available cycles. "
        
        if mem_val > 85:
            diagnostic += "CRITICAL: Memory pressure is extremely high, which may cause I/O wait times to climb regardless of CPU load."
        
        intelligence_report.append(diagnostic)

        # 3. Actionable Intelligence (The 'Gemini' Suggestions)
        if future_cpu > 75:
            advice = "ADVICE: Immediate intervention recommended. Please terminate high-impact background tasks. If this is a server, consider scaling or enabling aggressive throttling."
        elif mem_val > 80:
            advice = "ADVICE: Resource Leak Detected? Memory usage is disproportionate to CPU load. Consider restarting memory-intensive applications to reclaim heap space."
        else:
            advice = "ADVICE: Performance is healthy. This is an ideal window for running maintenance tasks or complex compilations."

        intelligence_report.append(advice)

        # =========================
        # STORE RESULT
        # =========================
        session["LAST_PREDICTION"] = {
            "time": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
            "future_cpu": round(future_cpu, 2),
            "state": state,
            "recommendations": intelligence_report
        }

        return jsonify({
            "user": request.user["email"],
            "future_cpu": round(future_cpu, 2),
            "state": state,
            "recommendations": intelligence_report
        })



    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route("/client-system", methods=["GET"])
@token_required
def client_system():
    user_email = request.user["email"]
    session = get_user_session(user_email)
    if session["LAST_SYSTEM_DATA"] is None:
        return jsonify({"status": "warming_up"})
    return jsonify({
        "current": session["LAST_SYSTEM_DATA"],
        "history": session["SYSTEM_HISTORY"]
    })
@app.route("/status")
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH,
        "model_error": MODEL_LOAD_ERROR
    })
@app.route("/predicted", methods=["GET"])
@token_required
def predicted():
    user_email = request.user["email"]
    session = get_user_session(user_email)
    if session["LAST_PREDICTION"] is None:
        return jsonify({
            "status": "warming_up",
            "message": "Waiting for prediction"
        })
    return jsonify(session["LAST_PREDICTION"])
@app.route("/delete-account", methods=["DELETE"])
@token_required
def delete_account():
    if users_collection is None:
        return jsonify({"error": "Database not connected"}), 500
    user_email = request.user["email"]
    try:
        result = users_collection.delete_one({"email": user_email})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), 404
        if user_email in USER_SESSIONS:
            del USER_SESSIONS[user_email]
        return jsonify({"status": "deleted", "message": "Account successfully terminated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print("Starting AI Resource Optimization Backend...")
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
