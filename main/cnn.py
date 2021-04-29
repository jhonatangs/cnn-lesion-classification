import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Layer,
    Input,
    Conv2D,
    ZeroPadding2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    Add,
)


class IdentityBlock(Layer):
    def __init__(self, kernel_size, filters, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.filters = filters

        f1, f2, f3 = filters
        self.layers = []

        self.layers.append(Conv2D(f1, (1, 1)))
        self.layers.append(BatchNormalization(axis=3))
        self.layers.append(Activation("relu"))

        self.layers.append(Conv2D(f2, kernel_size, padding="same"))
        self.layers.append(BatchNormalization(axis=3))
        self.layers.append(Activation("relu"))

        self.layers.append(Conv2D(f3, (1, 1)))
        self.layers.append(BatchNormalization(axis=3))

        self.layers.append(Add())
        self.layers.append(Activation("relu"))

    def call(self, X):
        X_shortcut = X

        for i in range(len(self.layers)):
            if i == len(self.layers) - 2:
                X = self.layers[i]([X, X_shortcut])
            else:
                X = self.layers[i](X)

        return X

    def get_config(self):
        config = super().get_config().copy()
        config.update({"kernel_size": self.kernel_size, "filters": self.filters})
        return config


class ConvolutionalBlock(Layer):
    def __init__(self, kernel_size, stride, filters, **kwargs):
        super(ConvolutionalBlock, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.filters = filters

        f1, f2, f3 = filters
        self.layers = []
        self.layers_shortcut = []

        self.layers.append(Conv2D(f1, (1, 1), (stride, stride)))
        self.layers.append(BatchNormalization(axis=3))
        self.layers.append(Activation("relu"))

        self.layers.append(Conv2D(f2, kernel_size, padding="same"))
        self.layers.append(BatchNormalization(axis=3))
        self.layers.append(Activation("relu"))

        self.layers.append(Conv2D(f3, (1, 1)))
        self.layers.append(BatchNormalization(axis=3))

        self.layers_shortcut.append(Conv2D(f3, (1, 1), (stride, stride)))
        self.layers_shortcut.append(BatchNormalization(axis=3))

        self.layers.append(Add())
        self.layers.append(Activation("relu"))

    def call(self, X):
        X_shortcut = X

        for layer in self.layers_shortcut:
            X_shortcut = layer(X_shortcut)

        for i in range(len(self.layers)):
            if i == len(self.layers) - 2:
                X = self.layers[i]([X, X_shortcut])
            else:
                X = self.layers[i](X)

        return X

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "filters": self.filters,
            }
        )
        return config


class ResNet50:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    def build(self):
        return Sequential(
            [
                Input(self.input_shape),
                ZeroPadding2D((3, 3)),
                Conv2D(64, (7, 7), (2, 2)),
                BatchNormalization(3),
                Activation("relu"),
                ZeroPadding2D(),
                MaxPool2D((3, 3), (2, 2)),
                ConvolutionalBlock(kernel_size=3, stride=1, filters=[64, 64, 256]),
                IdentityBlock(kernel_size=3, filters=[64, 64, 256]),
                IdentityBlock(kernel_size=3, filters=[64, 64, 256]),
                ConvolutionalBlock(kernel_size=3, stride=2, filters=[128, 128, 512]),
                IdentityBlock(kernel_size=3, filters=[128, 128, 512]),
                IdentityBlock(kernel_size=3, filters=[128, 128, 512]),
                IdentityBlock(kernel_size=3, filters=[128, 128, 512]),
                ConvolutionalBlock(kernel_size=3, stride=2, filters=[256, 256, 1024]),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024]),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024]),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024]),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024]),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024]),
                ConvolutionalBlock(kernel_size=3, stride=2, filters=[512, 512, 2048]),
                IdentityBlock(kernel_size=3, filters=[512, 512, 2048]),
                IdentityBlock(kernel_size=3, filters=[512, 512, 2048]),
                GlobalAveragePooling2D(),
                Dense(units=self.classes, activation="softmax"),
            ]
        )
