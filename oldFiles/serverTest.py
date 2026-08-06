import eventlet
eventlet.monkey_patch()

import socketio

sio = socketio.Server()
app = socketio.WSGIApp(sio)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 1)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        print("speed:", data.get('speed'))
        send_control(0.5, 1.0)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('disconnect')
def disconnect(sid):
    print("disconnect ", sid)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True
    )


if __name__ == '__main__':
    print("Starting server on port 4567")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)