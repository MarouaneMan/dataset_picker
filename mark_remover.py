import cv2
import numpy as np

class MarkRemover():

    def calc_ratio(self, img):
        # Calculate ratio
        ratio = img.shape[0] / img.shape[1]
        targetWidth = 600
        targetHeight = int(targetWidth * ratio)
        if targetHeight > 1024:
            targetHeight = 1024
            targetWidth = int(targetHeight / ratio)
        return (targetWidth, targetHeight)

    def white_blend(self, fg):
        bg = np.zeros((fg.shape[0], fg.shape[1], 3), np.uint8)
        bg[:] = 255
        output = np.empty((fg.shape[0], fg.shape[1], 3), np.uint8)
        alpha = fg[:,:,3] / 255.0
        output[:,:,0] = fg[:,:,0] * alpha + bg[:,:,0] * (1 - alpha)
        output[:,:,1] = fg[:,:,1] * alpha + bg[:,:,1] * (1 - alpha)
        output[:,:,2] = fg[:,:,2] * alpha + bg[:,:,2] * (1 - alpha)
        return output

    def show_image(self, image):
        targetWidth, targetHeight = self.calc_ratio(image)
        resized = cv2.resize(image, (targetWidth, targetHeight))
        cv2.imshow('Image', resized)
        pass

    def process_file(self, file, show=False):
        original = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = self.white_blend(original)

        # Convert to grayscale
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         

        # Invert black and white
        newRet, binary = cv2.threshold(imgGray, 254, 255,cv2.THRESH_BINARY_INV)
        for x in range(0, 4):
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            binary = cv2.drawContours(binary, contours, -1, (255, 255, 255), 3)

        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a bounding box around all contours
        valid_contours = []
        for (c, h) in zip(contours, hierarchy[0]):
            if h[3] == -1 :
                if (cv2.contourArea(c)) > 20:
                    valid_contours.append(c)
        
        # Sort contours
        valid_contours = sorted(valid_contours, key=cv2.contourArea)
        valid_contours = valid_contours[:-1]
        for c in valid_contours:
            cv2.fillPoly(original, pts=[c], color=(255, 0, 255, 0))

        if show:
            self.show_image(img)
            cv2.waitKeyEx()
        #return len(valid_contours)
        return original

# if __name__ == "__main__":
#     markRemover = MarkRemover()
#     markRemover.process_file("test_1.png", show=True)
#     log = open("contours.txt","w")
#     files = glob.glob("E:/dataset/deviantart/magic-pngs/**/*.png")
#     index = 0
#     while index < len(files):
#         file = files[index]
#         contours = process_file(file)
#         print(file, contours, index, "/", len(files))
#         if contours > 1:
#             log.write(file + "\n")
#         index = index + 1
