import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Links:
# https://ezgif.com/svg-to-png
# Draw map: https://editor.method.ac/
# Step 1: draw map
# Step 2: Download as svg and convert to two diff size imgs: (160x160) and (40x40)

def save_to_text(img, filename):
    text = ""
    H, W = img.shape
    for r in range(H):
        new_str = ["%d" % img[r][c] for c in range(W)]
        text += ','.join(new_str) + '\n'
    with open(filename, 'w+') as f:
        f.write(text)


map_num = 1
# store original size image as text for training
im = cv2.imread("Maps/map%d_png_version.png" % map_num)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = (im / np.max(im)).astype('uint8')
im = 1 - im
cv2.imshow('im', im *  255)
cv2.waitKey(0)
save_to_text(im, "Maps/map%d_orig.txt" % map_num)

# Store mini version 
im = cv2.imread("Maps/map%d_mini_version.png" % map_num)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = (im / np.max(im)).astype('uint8')
im = 1 - im
cv2.imshow('im', im *  255)
cv2.waitKey(0)
np.save("Maps/map%d_mini" % map_num, im)


