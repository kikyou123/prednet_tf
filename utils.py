"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
import scipy.misc
import numpy as np

def save_img(img, image_path):
    img = img * 255
    img = draw_frame(img, True)
    scipy.misc.imsave(image_path, img)

def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def save_images(images, size, image_path):
  return imsave(images*255., size, image_path)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    image = draw_frame(image, is_input=True)
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img


def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))




def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:2,:,0]  = img[:2,:,2] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,2] = 0 
    img[:,-2:,0] = img[:,-2:,2] = 0 
    img[:2,:,1]  = 255 
    img[:,:2,1]  = 255 
    img[-2:,:,1] = 255 
    img[:,-2:,1] = 255 
  else:
    img[:2,:,0]  = img[:2,:,1] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,1] = 0 
    img[:,-2:,0] = img[:,-2:,1] = 0 
    img[:2,:,2]  = 255 
    img[:,:2,2]  = 255 
    img[-2:,:,2] = 255 
    img[:,-2:,2] = 255 

  return img 

