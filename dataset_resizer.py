import cv2
import glob
import tqdm



if __name__ == "__main__":

    target = "E:\\generated_dataset\\"
    files = glob.glob(target + '\\**\\*.jpg', recursive=True)
    for file in tqdm.tqdm(files):
        new_path = file.replace("_dataset", "_dataset_small")
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (320, 320))
        cv2.imwrite(new_path, img)
