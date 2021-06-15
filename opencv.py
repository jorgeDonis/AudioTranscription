import numpy as np
import cv2

def show_img(img) -> None:
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("./MTD/data_AUDIO_IMG/MTD0429_Bach_BWV0866-01.png")
pad_img = np.pad(img, ( (0, 0), (0, 100), (0, 0) ), 'constant', constant_values= ( (0, 0), (0, 0), (0, 0) ))
