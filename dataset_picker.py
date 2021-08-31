import cv2
import os
import glob
import sys
import numpy as np

def blend(fg, bg):
    output = np.empty((fg.shape[0], fg.shape[1], 3), np.uint8)
    # Normalize alpha channel
    alpha = fg[:,:,3] / 255.0
    # output[:,:,0] = np.clip(fg[:,:,0] * alpha + bg[:,:,0] * (1 - alpha), 0, 255)
    # output[:,:,1] = np.clip(fg[:,:,1] * alpha + bg[:,:,1] * (1 - alpha), 0, 255)
    # output[:,:,2] = np.clip(fg[:,:,2] * alpha + bg[:,:,2] * (1 - alpha), 0, 255)
    output[:,:,0] = fg[:,:,0] * alpha + bg[:,:,0] * (1 - alpha)
    output[:,:,1] = fg[:,:,1] * alpha + bg[:,:,1] * (1 - alpha)
    output[:,:,2] = fg[:,:,2] * alpha + bg[:,:,2] * (1 - alpha)
    return output

def make_img(img, color):
    bg = np.full((img.shape[0], img.shape[1], 3), color, np.uint8)
    return blend(img, bg)

if __name__ == "__main__":
        
    target = sys.argv[1]

    for filename in glob.iglob(target + '**/**/*.*', recursive=True):
        
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        
        # Calculate ratio
        ratio = img.shape[0] / img.shape[1]
        targetWidth = 600
        targetHeight = int(targetWidth * ratio)
        img = cv2.resize(img, (targetWidth, targetHeight))
        
        cv2.imwrite('source.png', img)

        # Create mask
        bg_mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        fg_mask = img.copy()
        fg_mask[:,:,0:3] = 255
        mask = blend(fg_mask, bg_mask)

        # Background
        width = 64
        black_padded_img = np.pad(img, [(width, width), (width, width), (0,0)], mode='constant', constant_values=0)
        white_padded_img = np.pad(img, [(width, width), (width, width), (0,0)], mode='constant', constant_values=255)
        black_padded_mask = np.pad(mask, [(width, width), (width, width), (0,0)], mode='constant', constant_values=0)
        black = make_img(black_padded_img, (0,0,0))
        white = make_img(white_padded_img, (255,255,255))
        output = np.concatenate((black, white, black_padded_mask), axis=1)

        cv2.imshow('Image', output)
        key = cv2.waitKey(0)

        # #print(img[50:])
        # # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow('image', 1024, 1024)
        # print(filename)
