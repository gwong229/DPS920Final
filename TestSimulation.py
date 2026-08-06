import base64
import socketio
import eventlet
import eventlet.wsgi
import numpy as np
from io import BytesIO
from PIL import Image
from keras.models import load_model
from preProcessing import preprocess_image


# Socket.IO setup
sio = socketio.Server()

def app(environ, start_response):
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]

model = None

# Target speed the car will try to hold (mph)
MAX_SPEED = 20
MIN_SPEED = 10
speed_limit = MAX_SPEED

# fire every time the simulator sends a frame
@sio.on("telemetry")
def telemetry(sid, data):
    if data:
        # Current speed and center camera image from the sim
        speed = float(data["speed"])

        image_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(image_string)))
        image = np.asarray(image) 

        try:
            # Same preprocessing pipeline used during training
            processed = preprocess_image(image)
            processed = np.array([processed])
            steering_angle = float(model.predict(processed, verbose=0)[0][0])

            # slow down when going fast/turning hard,
            # speed back up when under the limit. Keeps throttle non-erratic.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED 
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            print(f"steering: {steering_angle:.4f}  throttle: {throttle:.4f}  speed: {speed:.2f}")
            send_control(steering_angle, throttle)

        except Exception as e:
            print(f"Prediction error: {e}")
    else:
        # No data  - sim disconnected/paused
        sio.emit("manual", data={}, skip_sid=True)

@sio.on("connect")
def connect(sid, environ):
    print("Connected:", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
        skip_sid=True,
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drive the Udacity simulator with a trained Keras model")
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default="selfDrivingModel.keras",
        help="Path to the trained .keras model file",
    )
    args = parser.parse_args()
    model = load_model(args.model)

    # Wrap our minimal app with socketio middleware
    app = socketio.Middleware(sio, app)

    # Serve on port 4567 (fixed port the Udacity sim expects)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)