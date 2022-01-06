import pickle
import numpy as np
import os
import lasagne
from PIL import Image
import torch
from torchvision import datasets, transforms


def main():
    in_path = "horse2zebra/train"
    out_path = "horse2zebra_64/"

    for filename in os.listdir(in_path):
        img = Image.open(os.path.join(in_path, filename))
        if img is not None:
            img = img.resize((64, 64))
            img.save(os.path.join(out_path, filename))

if __name__ == "__main__":
    main()