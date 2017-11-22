'''
Image Preprocessing script.
Converts image data to numpy array with each image as 128*128 in size
'''
from PIL import Image
import numpy as np
import os

def load_data(root):
    img = []
    label = []
    for image in os.listdir(root):
        im = Image.open(os.path.join(root,image))
        im = im.resize((128,128))
        img.append(np.array(im))
        label.append(image.split(".")[0])
    return np.array(img) , np.array(label)