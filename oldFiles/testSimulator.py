import eventlet
eventlet.monkey_patch()

import socketio
import numpy as np
import cv2
import base64

from keras.models import load_model
from preProcessing import preprocess_image


# ==========================
# Load model
# ==========================

print("Loading model...")
model = load_model("selfDrivingModel.keras")
print("Model loaded.")


# ==========================
# Socket.IO server
# ==========================

sio = socketio.Server(
    async_mode="eventlet",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False
)

app = socketio.WSGIApp(sio)


# ==========================
# Connection
# ==========================

@sio.event
def connect(sid, environ):
    print("Simulator connected:", sid)


@sio.event
def disconnect(sid):
    print("Simulator disconnected:", sid)


# ==========================
# Send steering commands
# ==========================

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        {
            "steering_angle": str(steering_angle),
            "throttle": str(throttle)
        }
    )
    print("Control sent:", steering_angle, throttle)


# ==========================
# Receive telemetry
# ==========================

@sio.on("telemetry", namespace="/")
def telemetry(sid, data):
    if data is None:
        return

    try:
        # ------------------
        # Decode image
        # ------------------
        image = data["image"]
        image = base64.b64decode(image)
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            print("Image decoding failed")
            return

        # ------------------
        # Preprocess
        # ------------------
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image = np.expand_dims(image, axis=0)

        # ------------------
        # Predict
        # ------------------
        steering_angle = float(model.predict(image, verbose=0)[0][0])
        print("Steering:", steering_angle)

        # ------------------
        # Control
        # ------------------
        send_control(steering_angle, 0.25)

    except Exception as e:
        print("Telemetry error:", e)


# ==========================
# Start server
# ==========================

if __name__ == "__main__":
    print("Starting server on port 4567")
    eventlet.wsgi.server(
        eventlet.listen(("", 4567)),
        app
    )