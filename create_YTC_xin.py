#!/usr/bin/env python
import numpy as np
from skimage.transform import resize
import h5py
import os
from PIL import Image
from time import time
import cv2 as cv
# import drawFromDataset

Num_img = 1640 #30976

def create_YTC(size):
    imgs = load_ytc_datasets(size)
    # idxs = np.random.permutation(np.arange(imgs.shape[0]))
    # imgs = imgs[idxs]
    print (imgs.shape)
    return imgs

def load_ytc_datasets(size):
    imgs = np.zeros((Num_img,3,size,size),dtype=np.float32)
    count = 0
    for i in range(1,Num_img+1):
        filename = r'./datasets/' + str(i) + '.png' ## input HR image path
        img = cv.imread(filename)
        # print (filename)
        # img = np.array(Image.open(filepath))
        # img = crop_image(img)
        # img = cv.resize(img,(32, 32),interpolation=cv.INTER_CUBIC)
        img = cv.resize(img, (size, size), interpolation=cv.INTER_CUBIC)/255.0
        # img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        img = np.transpose(img, (2, 0, 1))
        # imgs.append(img)
        if count >= Num_img:
            break
        imgs[count, :, :, :] = img
        count += 1

    # imgs = np.array(imgs, dtype=np.float32)
    # imgs = np.transpose(imgs, (0, 3, 1, 2))
    # print (imgs)
    return imgs

def crop_image(img):
    x, y , W, H = 35, 55, 100, 130
    img_new = img[y:(y+H), x:(x+W), :]
    return img_new

if __name__ == '__main__':
    start_time = time()
    x = create_YTC(18)
    end_time = time()
    # drawFromDataset.draw_save_images(x,10)
    print('Time using: %f'%(end_time-start_time))
    f = h5py.File('YTC_LR.hdf5', 'w')
    f.create_dataset('YTC', data=x)
    f.close()

    start_time = time()
    x = create_YTC(144)
    end_time = time()
    # drawFromDataset.draw_save_images(x,10)
    print('Time using: %f'%(end_time-start_time))
    f = h5py.File('YTC_HR.hdf5', 'w')
    f.create_dataset('YTC', data=x)
    f.close()

