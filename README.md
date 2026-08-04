# 🚗 Self-Driving Car Simulation (Behavioral Cloning)

This project teaches a car to drive itself, using nothing but images from its front
camera. We train a neural network to look at the road ahead and predict the correct
steering angle, then let it drive on its own inside a simulator.

It's built for DPS920, using the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim),
a free driving simulator originally built for Udacity's Self-Driving Car Nanodegree.

---

## 🧠 How It Works (the short version)

1. **We drive the car ourselves** in the simulator's Training Mode, and it records
   every camera frame along with the steering angle we used at that moment.
2. **We clean up that data** so the model doesn't just learn to drive straight all
   the time (most of any drive is spent going straight, so we trim that down a bit).
3. **We teach the model** using that data, showing it thousands of examples of what
   the road looked like and the correct steering angle for it.
4. **We let it drive on its own**, in the simulator's Autonomous Mode, using only what
   it learned.

---

## 📁 What's in This Repo

| File | What it does |
|---|---|
| `displayData.py` | Shows a chart of our collected steering data, so we can check it's not too lopsided. |
| `preProcessing.py` | Prepares images for the model: cropping, resizing, and randomly tweaking some images (flipped, brighter/darker, zoomed) so the model sees more variety. |
| `model.py` | Defines the actual neural network: what "shape" it is, how many layers, etc. |
| `fixPaths.py` | Fixes up file paths so the images can be found no matter whose computer we're running on. |
| `selfDrivingModel.py` | The main training script, this is what actually teaches the model to drive. |
| `testSimulator.py` | Connects our trained model to the simulator so it can drive live. |
| `serverTest.py` | An early test file we used just to check the simulator connection worked, before we had a real model. |

---

## ⚙️ Requirements

- [Python 3.10 or 3.11](https://www.python.org/downloads/) (newer versions like 3.12/3.13 aren't fully supported yet by some of the tools we use)
- The [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim/releases) (Term 1 version, pick the download for your operating system)
- [TensorFlow](https://www.tensorflow.org/install) and the other Python packages listed in `requirements.txt` (Windows) or `requirements-mac.txt` (Mac)

---

## 🛠️ Setup

**1. Create a virtual environment** (keeps this project's packages separate from everything else on your computer). This uses Python's built-in [venv module](https://docs.python.org/3/library/venv.html):

Windows:
```
py -3.10 -m venv venv
.\venv\Scripts\activate
```

Mac:
```
python3.10 -m venv venv
source venv/bin/activate
```

**2. Install the required packages:**
```
pip install -r requirements.txt
```
(Mac users on Apple Silicon: see the GPU note near the bottom for a couple of extra install steps.)

---

## ▶️ How to Run This Project

**Step 1: Collect driving data.**
Open the simulator, choose Training Mode, and drive a few laps around the track.
Click Record first so it saves your images and steering angles.

**Step 2: Check your data looks reasonable.**
```
python displayData.py
```

**Step 3: Fix up the file paths.**
```
python fixPaths.py
```

**Step 4: Train the model.**
```
python selfDrivingModel.py
```
This trains for a few minutes and saves a file called `selfDrivingModel.keras`, our trained model.

**Step 5: Watch it drive.**
```
python testSimulator.py
```
Then open the simulator again, but this time choose Autonomous Mode. The car should start driving on its own.

---

## 🍎 Note for Mac (Apple Silicon) Users

If you want your Mac's GPU to speed up training, install these two extra packages
after activating your virtual environment, using [Apple's tensorflow-metal plugin](https://developer.apple.com/metal/tensorflow-plugin/):
```
pip install tensorflow==2.18.1
pip install tensorflow-metal==1.2.0
```
We found that newer TensorFlow versions currently break Apple's GPU plugin, so we're
sticking with this specific combination for now.

---

## 🧩 Problems We Ran Into (and How We Fixed Them)

Every project has a few bumps along the way. Each of us is adding the issues we hit
personally, in case it helps someone else on the team (or anyone else following along).

### Riya's Issues

1. **My laptop's GPU couldn't actually be used for training.** I initially tried to
   set up GPU acceleration using [CUDA](https://developer.nvidia.com/cuda-toolkit),
   but CUDA only works with NVIDIA GPUs, and my laptop only has an Intel integrated
   graphics chip. There was no way around this on that machine, so I switched to
   training on CPU instead (and later moved to a Mac with Apple Silicon, which has
   its own separate GPU option, see the note above).
2. **Python 3.13 wasn't supported yet by TensorFlow.** My computer had a very new
   Python version installed by default, which caused install errors. I installed
   [Python 3.10](https://www.python.org/downloads/) separately and pointed my virtual
   environment at that version specifically.
3. **My Mac's GPU support broke after installing the newest TensorFlow.** It turned
   out the newest version wasn't compatible with Apple's GPU plugin yet, so I pinned
   to an older, working version instead.
4. **The simulator wouldn't open on my Mac** due to macOS's security settings blocking
   apps without a recognized developer signature. I fixed this with a one-time Terminal
   command that lets macOS trust it.

### [Add your name here]

_Add your own issues and fixes here!_

---
