# Self-Driving Car Simulation (Behavioral Cloning) - Project Documentation

## Approach
This project teaches a car to drive itself, using nothing but images from its front
camera. It implements trainin a CNN to steer the car autonomously,
in the Udacity simulator, looking at the road ahead and predicting the correct steering 
angle, then let it drive on its own. Training data was collected by manually driving
Track 1 in the simulator's training mode, whih logs the center camera image alongside the
corresponding steering angle, throttle, and speed for each frame.

Each image passes through a fixed preprocessing pipeline. It is cropped to isolate the road
surface, converted from RGB to YUV color space, resized to 200 x 66 to match the Nvidia
architecture's expected input, lightly blurred, and normalized. This same pipeline is
applied during training and live inference to guarantee consistency.

Additionally, data augmentation was used during training, randomly flipping (with steering
angle inverted to match), randomly translating horizontally and vertically with proportional
steering angle adjustment (simulating being off-center and thus needing to correct), randomly
adjusting brightness, and randomly zoomed. Of course, this is only applied to training data
and not validation data.

The model itself is the five-convolutional-layer, four-fully-connected-layer Nvidia architecture,
trained with the Adam optimizer and mean squared error loss between predicted and recorded
steering angle. Once trained, the saved model is loaded with the dirving script (TestSimulation.py)
that connects the simulator over Socket.IO, receives a live camera frame and speed reading with
every telemetry update, runs it through the same preprcessing pipeline, predicts a steering angle,
and computes throttle value. The predicted steering and throttle are then sent back to the simulator
in real time, closing the control loop.

---

## Major Challenges and How They Were Addressed

Bridging to an older simulator over Socket.IO was a big struggle for us. The Udacity simulator 
bundles an old JavaScript Socket.IO client, which is incompatible with the modern python-socketio/python-engineio 
libraries by default. Connections would appear to be accepted but the Socket.IO handshake itself would 
silently fail, so no telemetry ever arrived. This was fixed by pinning python-socketio and 
python-engineio to specific older, mutually compatible versions. Getting there also lead to a 
cascade of secondary dependency conflicts which we appropriately removed/replaced to fix.

Additionlly, the car drifted off-track despite driving smoothly at first. Our initial models followed the lane 
correctly for a while but would eventually drift and fail to recover. Since the project required 
center-camera-only training (no left/right camera correction), this was addressed by adding a random 
translation augmentation that synthetically shifts each training image and adjusts its steering angle 
proportionally to "teach" the model what an off-center view looks like and how much correction it 
calls for. Steering angle distribution was also examined and found to be heavily skewed toward near-zero 
(straight-line driving), so those samples were downsampled to reduce the model’s bias toward driving 
straight through curves.

Finally, was tuning through iteration. Getting from “technically drives” to “drives the course well” took a 
considerable amount of trial and error including: 1. increasing training epochs (an initial 10 was 
insufficient once richer augmentation was introduced; 60 produced better results) as well as adjusting augmentation 
parameters like the translation and zoom ranges.

### Other Problems We Ran Into and had to Overcome

1. **Some laptops GPU's couldn't actually be used for training.** I initially tried to
   set up GPU acceleration using [CUDA](https://developer.nvidia.com/cuda-toolkit),
   but CUDA only works with NVIDIA GPUs, and some devices only has an Intel integrated
   graphics chip. There was no way around this on that machine, so I switched to
   training on CPU instead (and later moved to a Mac with Apple Silicon, which has
   its own separate GPU option).
2. **Riya - Python 3.13 wasn't supported yet by TensorFlow.** My computer had a very new
   Python version installed by default, which caused install errors. I installed
   [Python 3.10](https://www.python.org/downloads/) separately and pointed my virtual
   environment at that version specifically.
3. **Riya - My Mac's GPU support broke after installing the newest TensorFlow.** It turned
   out the newest version wasn't compatible with Apple's GPU plugin yet, so I pinned
   to an older, working version instead.
4. **Riya - The simulator wouldn't open on my Mac** due to macOS's security settings blocking
   apps without a recognized developer signature. I fixed this with a one-time Terminal
   command that lets macOS trust it.

---

## What's in This Repo (The Important/Necessary Files)

| File | What it does |
|---|---|
| `displayData.py` | Shows a chart of our collected steering data, so we can check it's not too lopsided. |
| `preProcessing.py` | Prepares images for the model: cropping, resizing, and randomly tweaking some images (flipped, brighter/darker, zoomed) so the model sees more variety. |
| `model.py` | Defines the actual neural network: what "shape" it is, how many layers, etc. |
| `selfDrivingModel.py` | The main training script, this is what actually teaches the model to drive. |
| `TestSimulation.py` | Connects our trained model to the simulator so it can drive live. |
| `selfDrivingModel.keras` | Our final trained model that works successfully for autonomous driving. |

---

## Requirements

- [Python 3.10 or 3.11](https://www.python.org/downloads/) (newer versions like 3.12/3.13 aren't fully supported yet by some of the tools we use)
- The [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim/releases) (Term 1 version, pick the download for your operating system)
- [TensorFlow](https://www.tensorflow.org/install) and the other Python packages listed in `requirements.txt` (Windows) or `requirements-mac.txt` (Mac)

---

## How to Run This Project

### 1. Environment Setup

Because of the older Socket.IO version required to communicate with the simulator, dependencies must be installed at specific pinned versions:

```bash
pip install python-socketio==4.2.1 python-engineio==3.8.2
pip install eventlet dnspython --upgrade
pip install opencv-python numpy pandas scikit-learn pillow tensorflow
```

It's recommended to do this inside a dedicated virtual environment to avoid conflicts with other projects:

```bash
python -m venv sdc_env
sdc_env\Scripts\activate      # Windows
pip install <packages above>
```

### 2. Collect Training Data

Launch the Udacity simulator, choose Track 1, and select **Training Mode**. Drive several laps, then stop recording. This produces a `driving_log.csv` file and an `IMG/` folder of images inside your data directory (e.g. `data/`).

This produces `driving_log_fixed.csv` with paths rewritten to match your current `IMG/` folder location.

### 3. Train the Model

```bash
python selfDrivingModel.py
```

This loads the fixed CSV, splits it into training/validation sets, applies preprocessing and augmentation, trains the Nvidia-architecture model, and saves the result as `selfDrivingModel.keras`.

### 4. Test in the Simulator (Autonomous Mode)

```bash
python TestSimulation.py selfDrivingModel.keras
```

With the script running, open the simulator, select Track 1, and click **Autonomous Mode**. The simulator will connect to the running script over `localhost:4567`, and the trained model will take over steering and throttle control.


## Note for Mac (Apple Silicon) Users

If you want your Mac's GPU to speed up training, install these two extra packages
after activating your virtual environment, using [Apple's tensorflow-metal plugin](https://developer.apple.com/metal/tensorflow-plugin/):
```
pip install tensorflow==2.18.1
pip install tensorflow-metal==1.2.0
```
We found that newer TensorFlow versions currently break Apple's GPU plugin, so we're
sticking with this specific combination for now.

---