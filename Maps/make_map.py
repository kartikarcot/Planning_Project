import cv2
import numpy as np 
import matplotlib.pyplot as plt

def save_to_text(img, filename):
    text = ""
    H, W = img.shape
    for r in range(H):
        new_str = ["%d" % img[r][c] for c in range(W)]
        text += ','.join(new_str) + '\n'
    with open(filename, 'w+') as f:
        f.write(text)

# thres = 200
# im = cv2.imread("race_track2.png")

thresh = 50
# im = cv2.imread("race_track.jpg")
im = cv2.imread("map2.png")
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_im)
# plt.show()
mask = (gray_im < 200).astype("uint8")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.dilate(mask, kernel, iterations=1)
# plt.imshow(mask)
# plt.show()
cv2.imshow("mask", mask * 255)
cv2.waitKey(0)

np.save("map2", mask)
save_to_text(mask, "map2.txt")


