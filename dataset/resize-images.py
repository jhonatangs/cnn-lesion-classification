import os
import glob

from PIL import Image


def resize_images(path):
    classes = ["melanoma", "nevus", "seborrheic_keratosis"]

    for classe in classes:
        images = glob.glob(f"dataset/{path}/{classe}/*.jpg")

        for image in images:
            img = Image.open(image)
            img = img.resize((224, 224), Image.ANTIALIAS)
            image_name = image.split("/")[-1]
            img.save(f"dataset/{path}1/{classe}/{image_name}")


resize_images("Test")
resize_images("Training")
resize_images("Validation")