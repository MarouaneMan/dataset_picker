import os
import glob
#from albumentations.augmentations.geometric.functional import scale
import cv2
import numpy as np
import albumentations as A
from threading import Thread, Lock

output_folder = "synthetic_dataset"

def calc_new_dimensions(img):
    max_dim = 512
    width = img.shape[1]
    height = img.shape[0]
    ratio = width / height
    if width > max_dim:
        width = max_dim
        height = int(width / ratio)
    elif height > max_dim:
        height = max_dim
        width =  int(height * ratio)
    return (width, height)

def blend(fg, bg):
    output = np.empty((fg.shape[0], fg.shape[1], 3), np.uint8)
    # Normalize alpha channel
    alpha = fg[:,:,3] / 255.0
    output[:,:,0] = fg[:,:,0] * alpha + bg[:,:,0] * (1 - alpha)
    output[:,:,1] = fg[:,:,1] * alpha + bg[:,:,1] * (1 - alpha)
    output[:,:,2] = fg[:,:,2] * alpha + bg[:,:,2] * (1 - alpha)
    return output

def create_mask(img):
    bg_mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    fg_mask = img.copy()
    fg_mask[:,:,0:3] = 255
    mask = blend(fg_mask, bg_mask)
    return mask

def augment(img):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(
        #     shift_limit_x=0,
        #     shift_limit=0,
        #     shift_limit_y=0,
        #     scale_limit=0,
        #     rotate_limit=0,
        #     border_mode=cv2.BORDER_CONSTANT),
        A.Blur(p=0.05),
    ])

    # Augment an image
    transformed = transform(image=img)
    return transformed["image"]

def generate_img_mask(thread_id, dest_folder, index, fg_path, bg_list):
    
    # Read foreground
    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)

    # Calculate new dimensions
    #new_width, new_height = calc_new_dimensions(fg)
    new_width, new_height = (320, 320)

    # Read and resize the background
    bgs = []
    for bg_path in bg_list:
        bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        bg_resized = cv2.resize(bg, (new_width, new_height))
        bgs.append(bg_resized)

    # Resize foreground
    fg_resized = cv2.resize(fg, (new_width, new_height))

    # Blend
    output_primary = blend(fg_resized, bgs[0])
    mask_primary = create_mask(fg_resized)

    prefix = 'th' + str(thread_id) + '_' + str(index)

    # Image path
    image_path  = os.path.join(dest_folder, 'image')
    mask_path   = os.path.join(dest_folder, 'mask')

    # Write to disk
    cv2.imwrite(os.path.join(image_path, prefix + '_0.jpg'), output_primary)
    cv2.imwrite(os.path.join(mask_path, prefix + '_0.jpg'), mask_primary)

    for i in range(1, 2):
        fg_augmented = augment(fg_resized)
        mask_augmented = create_mask(fg_augmented)
        output_augmented = blend(fg_augmented, bgs[i])
        cv2.imwrite(os.path.join(image_path, prefix + '_' + str(i) + '.jpg'), output_augmented)
        cv2.imwrite(os.path.join(mask_path, prefix + '_' + str(i) + '.jpg'), mask_augmented)
    pass

bg_train_index = -1
bg_test_index = -1
bg_train_size = 0
bg_test_size = 0
bg_mutex_train = Lock()
bg_mutex_test = Lock()

def train_get_bg_index():
    global bg_train_index
    global bg_mutex_train
    global bg_train_size

    bg_mutex_train.acquire()
    bg_train_index = (bg_train_index + 1) % bg_train_size
    copy = bg_train_index
    bg_mutex_train.release()
    return copy

def test_get_bg_index():
    global bg_test_index
    global bg_mutex_test
    global bg_test_size

    bg_mutex_test.acquire()
    bg_test_index = (bg_test_index + 1) % bg_test_size
    copy = bg_test_index
    bg_mutex_test.release()
    return copy

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def train_worker(thread_id, fg_list, bg_list):
    index = 0
    while index < len(fg_list):
        try:
            fg_path = fg_list[index]
            bgs = []
            for _ in range(0,5):
                bgs.append(bg_list[train_get_bg_index()])
            generate_img_mask(thread_id, "E:/" + output_folder + "/train", index, fg_path, bgs)
        except:
            pass
        index = index + 1
    pass

def test_worker(thread_id, fg_list, bg_list):
    index = 0
    while index < len(fg_list):
        try:
            fg_path = fg_list[index]
            bgs = []
            for _ in range(0,5):
                bgs.append(bg_list[test_get_bg_index()])
            generate_img_mask(thread_id, "E:/" + output_folder + "/test", index, fg_path, bgs)
        except:
            pass
        index = index + 1
    pass

if __name__ == "__main__":

    fg_files        = glob.glob("E:/dataset/**/*.png", recursive=True)
    bg_files        = glob.glob("E:/unsplash/gallery-dl/unsplash/**/*.jpg", recursive=True)
    ratio           = 0.8
    fg_train_size   = int(ratio * len(fg_files))
    bg_train_size   = int(ratio * len(bg_files))
    fg_test_size    = len(fg_files) - fg_train_size
    bg_test_size    = len(bg_files) - bg_train_size
    fg_train        = fg_files[0:fg_train_size]
    bg_train        = bg_files[0:bg_train_size]
    fg_test         = fg_files[fg_train_size:]
    bg_test         = bg_files[bg_train_size:]

    print(len(fg_train), len(bg_train))
    print(len(fg_test), len(bg_test))

    train_threads = []
    train_index = 0
    for list in chunks(fg_train, int(len(fg_train) / 10)):
        t = Thread(target=train_worker, args=(train_index, list, bg_train))
        train_threads.append(t)
        t.start()
        print("Thread train ", train_index, " started", flush=True)
        train_index = train_index + 1
    
    # Wait train
    for th in train_threads:
        th.join()
        print("Thread train joined")
        
    test_threads = []
    test_index = 0
    for list in chunks(fg_test, int(len(fg_test) / 10)):
        t = Thread(target=test_worker, args=(test_index, list, bg_test))
        test_threads.append(t)
        t.start()
        print("Thread test ", test_index, " started", flush=True)
        test_index = test_index + 1
    
    # Wait test
    for th in test_threads:
        th.join()
        print("Thread test joined")
    pass