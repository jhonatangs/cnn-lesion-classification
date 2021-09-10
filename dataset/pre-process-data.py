import os
import shutil

import numpy as np
import pandas as pd


def find_one(arr):
    for i in range(len(arr)):
        if arr[i] == 1:
            return i


def create_dir(dir_name):
    os.chdir("dataset")
    ground_truth = pd.read_csv(f"ISIC-2017_{dir_name}_Part3_GroundTruth.csv")
    ground_truth = ground_truth.astype({"melanoma": int, "seborrheic_keratosis": int})
    nevus = []

    for _, i in ground_truth.iterrows():
        nevus.append(1 if i[1] == 0 and i[2] == 0 else 0)

    ground_truth["nevus"] = nevus

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    if os.path.exists(dir_name):
        os.chdir(dir_name)

    for column in ground_truth.columns[1:]:
        if not os.path.exists(column):
            os.mkdir(column)

    os.chdir(f"../ISIC-2017_{dir_name}_Data/")

    for _, i in ground_truth.iterrows():
        shutil.copy(
            f"{i[0]}.jpg",
            f"../{dir_name}/{ground_truth.columns[find_one(i.values)]}",
        )

    os.chdir("../../")


def main():
    create_dir("Training")
    create_dir("Test")
    create_dir("Validation")


if __name__ == "__main__":
    main()
