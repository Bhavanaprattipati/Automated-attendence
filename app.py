import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session
import insightface

# ---------------------------
# Config
# ---------------------------
STUDENTS_DIR = "students"
UPLOADS_DIR = "uploads"
PROCESSED_DIR = os.path.join("static", "processed")
LOGS_DIR = "logs"

THRESHOLD = 0.50
SECRET_KEY = "supersecretkey"   # change in production

for d in (STUDENTS_DIR, UPLOADS_DIR, PROCESSED_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------
# Init app + Face model
# ---------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

face_app = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


# ---------------------------
# Helpers
# ---------------------------
def normalize(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def load_known_faces():
    known = {}
    files = sorted(os.listdir(STUDENTS_DIR))
    for fname in files:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(STUDENTS_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = face_app.get(img)
        if not faces:
            continue
        emb = faces[0].embedding
        embn = normalize(emb)
        name = os.path.splitext(fname)[0].split('_')[0]
        known.setdefault(name, []).append(embn)
    return known

def save_processed_image(img_bgr):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"processed_{ts}.jpg"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, img_bgr)
    return out_name

def log_attendance(present_names):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    logfile = os.path.join(LOGS_DIR, f"attendance_{date_str}.csv")
    if os.path.exists(logfile):
        df = pd.read_csv(logfile)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    changed = False
    for name in present_names:
        if not ((df["Name"] == name).any()):
            df.loc[len(df)] = [name, date_str, time_str]
            changed = True

    if changed:
        df.to_csv(logfile, index=False)
    return logfile


# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email.endswith("@gvpce.ac.in") and password.endswith("@123"):
            session["user"] = email
            return redirect(url_for("attendance_page"))
        else:
            return render_template("login.html", error="Invalid credentials. Try again.")
    return render_template("login.html")


@app.route("/attendance", methods=["GET"])
def attendance_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", processed_image=None, attendance=None, threshold=THRESHOLD)


@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    known_faces = load_known_faces()
    file = request.files.get("file")

    if not file or file.filename == "":
        return render_template("index.html", message="No file selected", attendance=None, processed_image=None)

    filename = file.filename
    save_path = os.path.join(UPLOADS_DIR, filename)
    file.save(save_path)

    img_bgr = cv2.imread(save_path)
    if img_bgr is None:
        return render_template("index.html", message="Error reading image", attendance=None, processed_image=None)

    faces = face_app.get(img_bgr)
    if not faces:
        return render_template("index.html", message="No faces detected", attendance={}, processed_image=None)

    attendance = {name: "Absent" for name in known_faces.keys()}
    present_set = set()

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        embn = normalize(face.embedding)

        best_name, best_sim = None, -1.0
        for name, emb_list in known_faces.items():
            sims = [float(np.dot(embn, e)) for e in emb_list]
            if sims and max(sims) > best_sim:
                best_sim = max(sims)
                best_name = name

        if best_name and best_sim >= THRESHOLD:
            label = f"{best_name} ({best_sim:.2f})"
            attendance[best_name] = "Present"
            present_set.add(best_name)
            color = (0, 200, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, label, (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    processed_name = save_processed_image(img_bgr)
    logfile = log_attendance(list(present_set))

    return render_template("index.html", processed_image=processed_name, attendance=attendance, threshold=THRESHOLD, logfile=logfile)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
