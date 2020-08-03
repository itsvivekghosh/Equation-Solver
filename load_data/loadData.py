# Importing Important Libraries
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import pandas as pd

# Loading Images from the Specific Folder
def load_images_from_folder(folder):

    train_data = []

    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = ~img

        if img is not None:

            ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            ctrs,ret = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w = int(28)
            h = int(28)
            maxi = 0

            for c in cnt:
                x, y, w, h = cv2.boundingRect(c)
                maxi = max(w * h, maxi)
                if maxi == w * h:
                    x_max = x
                    y_max = y
                    w_max = w
                    h_max = h

            im_crop = thresh[y_max : y_max + h_max + 10, x_max : x_max + w_max + 10]
            im_resize = cv2.resize(im_crop, (28, 28))
            im_resize = np.reshape(im_resize, (784, 1))
            train_data.append(im_resize)

    return train_data


def saveDataset(data, save_dir):
    df = pd.DataFrame(data, index=None)
    file_name = "train_final.csv"
    df.to_csv("{}/{}".format(save_dir, file_name), index=False)
    print("Our Dataset is Saved as {}!".format(file_name))


def load_images(dataset_dir):
    dataset = []
    folders = os.listdir(dataset_dir)

    for folder_name in folders:
        print(dataset_dir, folder_name)
        data = load_images_from_folder(dataset_dir + '/' + folder_name)

        if folder_name == "+":
            for i in range(len(data)):
                data[i] = np.append(data[i], ["11"])

        elif folder_name == "-":
            for i in range(len(data)):
                data[i] = np.append(data[i], ["10"])

        elif folder_name == "times":
            for i in range(len(data)):
                data[i] = np.append(data[i], ["12"])

        else:
            for i in range(len(data)):
                data[i] = np.append(data[i], [folder_name])

        if folder_name == "+":
            dataset = data

        else:
            dataset = np.concatenate((dataset, data))

        print("The Total No. of Images in {} is: {}".format(folder_name, len(dataset)))

    return dataset


def loadAndSaveImages(dataset_dir="../dataset", save_dir="../train_data"):
    data = load_images(dataset_dir)
    print("Total Images are: " + str(len(data)))
    saveDataset(data, save_dir)


def main():
    loadAndSaveImages()


if __name__ == "__main__":
    main()
