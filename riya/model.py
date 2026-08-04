from keras import Sequential, layers
from keras.optimizers import Adam


def create_model():
    model = Sequential()

    # Input: 3@66x200
    # Output: 24@31x98
    model.add(
        layers.Conv2D(
            24,
            (5, 5),
            strides=(2, 2),
            activation="relu",
            input_shape=(66, 200, 3)
        )
    )

    # Output: 36 @ 14x47
    model.add(
        layers.Conv2D(
            36,
            (5, 5),
            strides=(2, 2),
            activation="relu"
        )
    )

    # Output: 48 @ 5x22
    model.add(
        layers.Conv2D(
            48,
            (5, 5),
            strides=(2, 2),
            activation="relu"
        )
    )

    # Output: 64 @ 3x20
    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation="relu"
        )
    )

    # Output: 64 @ 1x18
    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu"
        )
    )

    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(1164, activation="relu"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(10, activation="relu"))

    # Output vehicle control (steering angle)
    model.add(layers.Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="mse"
    )

    return model
