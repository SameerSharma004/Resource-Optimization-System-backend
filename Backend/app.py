from flask import Flask, jsonify, request, send_from_directory
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

app = Flask(__name__)
CORS(app)

SECRET_KEY = os.getenv("SECRET_KEY")
MONGO_URI = os.getenv("MONGO_URI","mongodb+srv://sam18112k4_db_user:lfZ3QKvlgznCpmTU@rosuserdata.pio2vka.mongodb.net/?appName=rosuserdata")
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "downloads")

users_collection = None

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()

    db = mongo_client["ros"]
    users_collection = db["userdata"]

    print("MongoDB connected successfully")

except Exception as e:
    print("MongoDB connection failed:", e)

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

scaler = MinMaxScaler()
scaler.fit([[0,0,0,0,0,0], [100,100,100,100,100,100]])

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
        "created_at": datetime.datetime.utcnow()
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
        return jsonify({"error": "Missing email or password"}), 400

    user = users_collection.find_one({"email": email})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid login"}), 401

    token = jwt.encode({
        "user_id": str(user["_id"]),
        "email": user["email"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
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
    session = get_user_session(user_email)

    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    # Store latest system data
    session["LAST_SYSTEM_DATA"] = data

    session["SYSTEM_HISTORY"].append({
        "time": datetime.datetime.utcnow().strftime("%H:%M:%S"),
        "cpu": data.get("cpu_usage", 0),
        "ram": data.get("memory_usage", 0),
        "net_up": data.get("net_upload_mbps", 0),
        "net_down": data.get("net_download_mbps", 0)
    })

    if len(session["SYSTEM_HISTORY"]) > 30:
        session["SYSTEM_HISTORY"].pop(0)

    try:

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

        session["FEATURE_BUFFER"].append(raw_features)
        if len(session["FEATURE_BUFFER"]) > SEQUENCE_LENGTH:
            session["FEATURE_BUFFER"].pop(0)

        if len(session["FEATURE_BUFFER"]) < SEQUENCE_LENGTH:
            padded_features = [session["FEATURE_BUFFER"][0]] * (SEQUENCE_LENGTH - len(session["FEATURE_BUFFER"])) + session["FEATURE_BUFFER"]
        else:
            padded_features = session["FEATURE_BUFFER"]

        scaled_features = scaler.transform(padded_features)

        sequence = np.array(scaled_features).reshape(
            1, SEQUENCE_LENGTH, len(raw_features)
        )

        prediction = model.predict(sequence, verbose=0)[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class])
        
        state_mapping = {
            0: "Idle",
            1: "Low Load",
            2: "Normal Load",
            3: "High Load",
            4: "Critical Load"
        }
        
        state = state_mapping.get(predicted_class, "Unknown")

        if predicted_class == 0:
            recommendations = [
                "System is currently idle. Consider enabling power saver mode.",
                "Reduce screen brightness to save battery life.",
                "Background processes are minimal."
            ]
        elif predicted_class == 1:
            recommendations = [
                "System is experiencing low load. Performance is optimal.",
                "You can comfortably run background tasks like updates or backups."
            ]
        elif predicted_class == 2:
            recommendations = [
                "System is under normal load with balanced resource usage.",
                "Operating efficiently. No immediate action needed."
            ]
        elif predicted_class == 3:
            recommendations = [
                "High resource utilization detected. System is under heavy load.",
                "Monitor your application usage to prevent lag."
            ]
            if cpu_val > 75:
                recommendations.append("CPU usage is high (>75%). Consider closing CPU-intensive applications.")
            if mem_val > 75:
                recommendations.append("Memory usage is high (>75%). Close unused tabs or programs.")
        elif predicted_class == 4:
            recommendations = [
                "CRITICAL LOAD: System is severely stressed and may become unresponsive.",
                "Immediately terminate non-essential heavy applications.",
                "Check for rogue processes causing high CPU or memory spikes."
            ]
        else:
            recommendations = ["System running optimally."]

        session["LAST_PREDICTION"] = {
            "time": datetime.datetime.utcnow().strftime("%H:%M:%S"),
            "idle_probability": round(float(prediction[0]), 2), # Keep backward compatibility by passing class 0 prob
            "prediction_confidence": round(confidence, 2),
            "state": state,
            "recommendations": recommendations
        }

        return jsonify({
            "user": request.user["email"],
            "idle_probability": round(float(prediction[0]), 2),
            "prediction_confidence": round(confidence, 2),
            "state": state,
            "recommendations": recommendations
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

if __name__ == "__main__":

    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print("Starting AI Resource Optimization Backend...")
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)