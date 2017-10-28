import cv2
import ssim
import numpy as np
from PIL import Image

target = np.zeros((120, 120, 3), np.uint8)
pred = np.zeros((120, 120, 3), np.uint8)

out = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_RGB2BGR)), Image.fromarray(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)))
print out
