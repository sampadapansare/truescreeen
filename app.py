from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import threading
import time

app = Flask(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ROBOFLOW_API_KEY = "ATCth3RHKPljJdY3UmHL"
ROBOFLOW_MODEL_ID = "interview-dxisb/3"

# Thresholds (# of consecutive frames)
ABSENCE_THRESHOLD = 10
INTRUDER_THRESHOLD = 10
ATTENTION_THRESHOLD = 15

# ─── GLOBALS ──────────────────────────────────────────────────────────────────

alert_message = ""
_last_alert_message = ""
lock = threading.Lock()
detection_result = {}
frame_for_detection = None

# ─── DETECTORS ────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# ─── BACKGROUND ROBOFLOW THREAD ───────────────────────────────────────────────

def roboflow_loop():
global detection_result, frame_for_detection
while True:
if frame_for_detection is not None:
resized = cv2.resize(frame_for_detection, (640, 640))
_, img_encoded = cv2.imencode('.jpg', resized)
try:
resp = requests.post(
f"https://detect.roboflow.com/{'interview-dxisb/3'}",
files={"file": img_encoded.tobytes()},
params={"api_key": 'ATCth3RHKPljJdY3UmHL',
"confidence": 60,
"overlap": 30}
)
with lock:
detection_result = resp.json()
except Exception as e:
print("Roboflow error:", e)
time.sleep(4)

threading.Thread(target=roboflow_loop, daemon=True).start()

# ─── VIDEO & FRAUD DETECTION ──────────────────────────────────────────────────

def generate_frames():
global alert_message, _last_alert_message, frame_for_detection

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

absence_timer = 0
intruder_timer = 0
attention_timer = 0

while True:
success, frame = cap.read()
if not success:
break

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
face_count = len(faces)

# hand off to Roboflow thread
frame_for_detection = frame.copy()

suspicious_object = False
for (x, y, w, h) in faces:
# draw face box
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# attention (eye) check
roi_gray = gray[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
if len(eyes) >= 2:
attention_timer = 0
else:
attention_timer += 1

# absence / intruder timers
absence_timer = absence_timer + 1 if face_count == 0 else 0
intruder_timer = intruder_timer + 1 if face_count > 1 else max(0, intruder_timer - 1)

# Roboflow object detection
with lock:
for obj in detection_result.get("predictions", []):
c = obj["confidence"]
area = obj["width"] * obj["height"]
if c >= 0.85 and area >= 5000:
suspicious_object = True
x, y = int(obj["x"]), int(obj["y"])
w, h = int(obj["width"]), int(obj["height"])
x1, y1 = x - w//2, y - h//2
x2, y2 = x + w//2, y + h//2
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.putText(frame,
f"{obj['class']} ({int(c*100)}%)",
(x1, y1 - 10),
cv2.FONT_HERSHEY_SIMPLEX,
0.6, (0, 0, 255), 2)
break

# decide final alert by priority
candidate = ""
if intruder_timer >= INTRUDER_THRESHOLD:
candidate = "🚨 Intruder Detected"
elif absence_timer >= ABSENCE_THRESHOLD:
candidate = "⚠️ No Person Detected"
elif suspicious_object:
candidate = "⚠️ Suspicious Object Detected"
elif attention_timer >= ATTENTION_THRESHOLD:
candidate = "⚠️ Attention Lost (Looking Away)"

# suppress repeats
if candidate != _last_alert_message:
alert_message = candidate
_last_alert_message = candidate

# encode & yield MJPEG frame
_, buf = cv2.imencode('.jpg', frame)
jpg = buf.tobytes()
yield (b'--frame\r\n'
b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

cap.release()

# ─── FLASK ROUTES ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
return render_template('index.html')

@app.route('/video_feed')
def video_feed():
return Response(generate_frames(),
mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status')
def alert_status():
return jsonify({"message": alert_message})

if __name__ == '__main__':
app.run(debug=True)
