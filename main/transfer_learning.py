import datetime
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout


def get_date_str():
    return str("{date:%d_%m_%Y_%H_%M}").format(date=datetime.datetime.now())


def get_time_min(start, end):
    return (end - start) / 60


EPOCHS = 200
FINE_TUNE_EPOCHS = 400
BATCH_SIZE = 64
INPUT_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
CLASSESS = 3
FILE_NAME = "cnn_model_lesion_"

print("\n\n ----------------------INICIO --------------------------\n")
print("[INFO] [INICIO]: " + get_date_str())
print("[INFO] Download dataset usando keras.preprocessing.image.ImageDataGenerator")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    "dataset/Training",
    target_size=INPUT_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="categorical",
)

test_set = test_datagen.flow_from_directory(
    "dataset/Test",
    target_size=INPUT_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="categorical",
)

validation_set = validation_datagen.flow_from_directory(
    "dataset/Validation",
    target_size=INPUT_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="categorical",
)

print("[INFO] Inicializando e otimizando a CNN...")
start = time.time()


base_model = ResNet152V2(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)

base_model.trainable = False

global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(CLASSESS, activation="softmax")

inputs = Input(shape=INPUT_SHAPE)
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = Dropout(0.35)(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)

base_learning_rate = 0.01

model.compile(
    optimizer=SGD(base_learning_rate, momentum=0.9, nesterov=True),
    loss="categorical_crossentropy",
    metrics=["accuracy", "AUC"],
)

print("[INFO] Treinando a CNN...")
history = model.fit(
    training_set,
    epochs=EPOCHS,
    validation_data=validation_set,
)

base_model.trainable = True

fine_tune_at = 300

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=SGD(base_learning_rate / 10, momentum=0.9, nesterov=True),
    loss="categorical_crossentropy",
    metrics=["accuracy", "AUC"],
)

history_fine = model.fit(
    training_set,
    epochs=FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=validation_set,
)

evaluate_result = model.evaluate(test_set, verbose=2)
print("EVALUATE:")
print(model.metrics_names)
print(evaluate_result)

print("[INFO] Salvando modelo treinado ...")

file_date = get_date_str()
model.save("models/" + FILE_NAME + file_date + ".h5")
print("[INFO] modelo: models/" + FILE_NAME + file_date + ".h5 salvo!")

end = time.time()

print("[INFO] Tempo de execução da CNN: %.1f min" % (get_time_min(start, end)))

print("[INFO] Summary: ")
model.summary()

print("\n[INFO] Avaliando a CNN...")
score = model.evaluate_generator(
    generator=test_set, steps=(test_set.n // test_set.batch_size), verbose=1
)
print("[INFO] Accuracy: %.2f%%" % (score[1] * 100), "| Loss: %.5f" % (score[0]))

print("[INFO] Gerando imagem do modelo de camadas da CNN")
plot_model(
    model, to_file="models/image/" + FILE_NAME + file_date + ".png", show_shapes=True
)

print("\n[INFO] [FIM]: " + get_date_str())
print("\n\n")
