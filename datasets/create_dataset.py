import pickle
import numpy as np
import os
import lasagne
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_folder, idx, img_size=32):

    d = unpickle(data_folder)
    x = d['data']
    y = d['labels']

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]


    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'))


def main():
    dict = load_databatch("val_data", 0, 64)
    x = dict["X_train"]
    y = dict["Y_train"]
    x = np.transpose(x, (0, 2, 3, 1))
    print(y)
    print(x.shape)
    # x = ((x + 1) * 127.5).clip(0, 255).astype(np.uint8)
    x = x.astype(np.uint8)
    x = x[y == 679, ...]
    y = y[y == 679]
    for i in range(y.shape[0]):
        im = Image.fromarray(x[i, ...])
        im.save(f"./imagenet64/{821}_{i}.jpg")

if __name__ == "__main__":
    main()