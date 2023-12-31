import os
import cv2
import time
import numpy as np
import albumentations as A
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


def resize_with_ratio(img, size):
    w, h = img.size
    if w > h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    return np.asarray(img.resize((new_w, new_h)))


def generate():
    folder = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g",
              "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

    counter = 0
    f_index = 0
    running = True
    transform = A.Compose([
        A.PadIfNeeded(336, 336, border_mode=cv2.BORDER_CONSTANT, value=0),
    ])

    while running:
        success, img = cap.read()
        hands, img = detector.findHands(img, flipType=False, draw=False)  # with draw
        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                hand_crop = img[y-30:y+h+30, x-30:x+w+30]
                resize_img = resize_with_ratio(Image.fromarray(hand_crop), 336)
                # cv2.imshow("Hand", hand_crop)
                trans_img = transform(image=resize_img)['image']
                cv2.imshow("Transform hand", trans_img)
            except:
                pass

        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            print(f"Save {counter} image")
            if not os.path.exists(f"generate/{folder[f_index]}"):
                os.makedirs(f"generate/{folder[f_index]}")
            cv2.imwrite(f"generate/{folder[f_index]}/{time.time()}.jpg", trans_img)

            if counter == 200:
                time.sleep(2)
                counter = 0
                f_index += 1
                if f_index == len(folder):
                    running = False
                else:
                    print(f"Change to Folder {folder[f_index]}")


if __name__ == '__main__':
    generate()
