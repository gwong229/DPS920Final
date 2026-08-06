import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preProcessing import augment, preprocess_image
from model import create_model

# Load dataset
data = pd.read_csv(
    "data/driving_log_fixed.csv",
    header=None
)

# Balance dataset: normal driving logs are dominated by near-zero steering angles
# which biases the model to jsut go straight. It hurts turning, so we need to
# downsample rows w/ small steering angle.
angles = data[3].astype(float)
near_zero = data[angles.abs() < 0.05]
rest = data[angles.abs() >= 0.05]

if len(near_zero) > 0:
    near_zero = near_zero.sample(frac=0.4, random_state=42)

data = pd.concat([near_zero, rest]).reset_index(drop=True)
print(f"Dataset size after balancing: {len(data)} rows")

# Split dataset
train_data, val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    # Convert BGR (OpenCV) to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image

def batch_generator(data, batch_size=32, training=True):
    while True:
        # Shuffle every epoch
        data = data.sample(
            frac=1
        ).reset_index(drop=True)

        for start in range(
            0,
            len(data),
            batch_size
        ):

            batch = data.iloc[start:start + batch_size]
            images = []
            steering_angles = []

            for i in range(len(batch)):

                # Image path from fixed CSV
                path = batch.iloc[i, 0]

                # Steering value
                angle = float(batch.iloc[i, 3])

                # Load image
                image = load_image(path)

                # Augmentation only for training data
                if training:
                    image, angle = augment(
                        image,
                        angle
                    )

                # Preprocessing:
                # - crop road area, RGB to YUV, resize to 200x66, normalize
                image = preprocess_image(image)

                images.append(image)
                steering_angles.append(angle)

            yield (
                np.array(images),
                np.array(steering_angles)
            )

# Create generators
train_generator = batch_generator(
    train_data,
    batch_size=32,
    training=True
)

val_generator = batch_generator(
    val_data,
    batch_size=32,
    training=False
)

# Create model
model = create_model()
model.summary()

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=len(train_data) // 32,
    validation_steps=len(val_data) // 32,
    epochs=60
)

# Save trained model
model.save("selfDrivingModel.keras")

# Print results
print("Training complete")
print(
    "Final training loss:",
    history.history["loss"][-1]
)

print(
    "Final validation loss:",
    history.history["val_loss"][-1]
)