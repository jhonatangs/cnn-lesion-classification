import datetime
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from cnn import ResNet50


def get_date_str():
    return str("{date:%d_%m_%Y_%H_%M}").format(date=datetime.datetime.now())


def get_time_min(start, end):
    return (end - start) / 60


EPOCHS = 400
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

early_stopping_monitor = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=15
)

model = ResNet50(INPUT_SHAPE, CLASSESS)
model = model.build()
model.compile(
    optimizer=SGD(momentum=0.9, nesterov=True),
    loss="categorical_crossentropy",
    metrics=["accuracy", "AUC"],
)

print("[INFO] Treinando a CNN...")
classifier = model.fit_generator(
    training_set,
    training_set.n // training_set.batch_size,
    epochs=EPOCHS,
    validation_data=validation_set,
    validation_steps=(validation_set.n // validation_set.batch_size),
    verbose=2,
    callbacks=[early_stopping_monitor],
)

EPOCHS = len(classifier.history["loss"])

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
