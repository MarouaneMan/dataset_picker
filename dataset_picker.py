import cv2
import os
import glob
import sys
import numpy as np
import sys
import shutil
from dataclasses import dataclass
from mark_remover import MarkRemover

KEY_ESCAPE      = 0x1b
KEY_ARROW_LEFT  = 0x250000
KEY_ARROW_RIGHT = 0x270000
KEY_ENTER       = 0xd
KEY_DELETE      = 0x2e0000
KEY_SPACE       = 0x20

@dataclass
class Feature:
    path: str
    deleted: bool
    mark_removed: bool

def calc_ratio(img):
    # Calculate ratio
    ratio = img.shape[0] / img.shape[1]
    targetWidth = 600
    targetHeight = int(targetWidth * ratio)
    if targetHeight > 1024:
        targetHeight = 1024
        targetWidth = int(targetHeight / ratio)
    return (targetWidth, targetHeight)

def create_mask(img):
    bg_mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    fg_mask = img.copy()
    fg_mask[:,:,0:3] = 255
    mask = blend(fg_mask, bg_mask)
    return mask

def blend(fg, bg):
    output = np.empty((fg.shape[0], fg.shape[1], 3), np.uint8)
    # Normalize alpha channel
    alpha = fg[:,:,3] / 255.0
    output[:,:,0] = fg[:,:,0] * alpha + bg[:,:,0] * (1 - alpha)
    output[:,:,1] = fg[:,:,1] * alpha + bg[:,:,1] * (1 - alpha)
    output[:,:,2] = fg[:,:,2] * alpha + bg[:,:,2] * (1 - alpha)
    return output

def make_img(img, color):
    bg = np.full((img.shape[0], img.shape[1], 3), color, np.uint8)
    return blend(img, bg)

if __name__ == "__main__":
        
    target = sys.argv[1]

    files = [Feature(file_path, False, False) for file_path in glob.glob(target + '**/**/*.*', recursive=True)]
    index = 0
    while index >= 0 and index < len(files):
        file        = files[index]
        print(file.path, "|", index, "/", len(files))
        filename    = os.path.basename(file.path)
        trash_path  = os.path.join('trash', filename)
        mark_path   = os.path.join('trash_mark', filename)
        file_path = file.path
        if file.deleted:
            file_path = trash_path
        
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        targetWidth, targetHeight = calc_ratio(img)
        img = cv2.resize(img, (targetWidth, targetHeight))
        
        # Create mask
        mask = create_mask(img)

        # Background
        width = 64
        black_padded_img = np.pad(img, [(width, width), (width, width), (0,0)], mode='constant', constant_values=0)
        white_padded_img = np.pad(img, [(width, width), (width, width), (0,0)], mode='constant', constant_values=255)
        black_padded_mask = np.pad(mask, [(width, width), (width, width), (0,0)], mode='constant', constant_values=0)
        black = make_img(black_padded_img, (0,0,0))
        white = make_img(white_padded_img, (255,255,255))
        output = np.concatenate((black, white, black_padded_mask), axis=1)

        cv2.imshow('Image', output)
        key = cv2.waitKeyEx(0)
        
        if key == -1 or key == KEY_ESCAPE: # Quit
            break
        elif key == KEY_ENTER: # Restore picture
            if file.deleted == True:
                shutil.move(trash_path, file.path)
                file.deleted = False
            index = index + 1
        elif key == KEY_DELETE: # Remove picture
            if file.deleted == False:
                shutil.move(file.path, trash_path)
                file.deleted = True
            index = index + 1
        elif key == KEY_ARROW_RIGHT:
            index = index + 1
        elif key == KEY_ARROW_LEFT:
            index = index - 1
        elif key == KEY_SPACE: # WaterMark removal
            if file.mark_removed == False:
                mark_remover = MarkRemover()
                cleaned = mark_remover.process_file(file_path)
                shutil.move(file.path, mark_path)
                cv2.imwrite(file.path, cleaned)
                file.mark_removed = True
            elif file.mark_removed == True:
                os.remove(file.path)
                shutil.move(mark_path, file.path)
                file.mark_removed = False
        
        index = 0 if index < 0 else index
        index = len(files) - 1 if index == len(files) else index
        sys.stdout.flush()
