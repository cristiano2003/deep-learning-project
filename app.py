import cv2
import torch
import json
import numpy as np
import albumentations as A
from cvzone.HandTrackingModule import HandDetector
from dl_project.model import ASLModel
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


@torch.no_grad()
def demo():
    model = ASLModel.load_from_checkpoint("./checkpoints/demo/resnet.ckpt",
                                          model="resnet", map_location='cpu')
    model.eval()
    transform = A.Compose([
        A.PadIfNeeded(112, 112, border_mode=cv2.BORDER_CONSTANT, value=0),
    ])
    with open('labels.json', 'r') as f:
        labels = json.load(f)
    invert_labels = {str(v): k for k, v in labels.items()}
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img, flipType=False, draw=False)  # with draw

        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                hand_crop = img[y-30:y+h+30, x-30:x+w+30]
                resize_img = resize_with_ratio(Image.fromarray(hand_crop).convert('L'), 112)
                # cv2.imshow("Hand", hand_crop)
                trans_img = transform(image=resize_img)['image'] / 255.
                cv2.imshow("trans", trans_img)
                label_idx = model(torch.from_numpy(trans_img).float().unsqueeze(0).unsqueeze(0)).argmax(-1).item()
                print("Output Label: ", invert_labels[str(label_idx)])

            except:
                pass

        cv2.waitKey(1)


if __name__ == '__main__':
    demo()
