import time
import tensorflow as tf
import skimage.color as color
import numpy as np
from collections import defaultdict
from PIL import Image
import os

def prep_for_inception(input_tensor):
    """
    convert rgb image to 3-layer grayscale image for Feature Extractor
    """
    res = tf.image.rgb_to_grayscale(np.asarray(input_tensor)) # convert rgb to grayscale
    res = tf.image.grayscale_to_rgb(res) # stack 3 layers of grayscale image
    res /= 255  # range[0,1] 
    res = tf.reshape(res, [-1, 299, 299, 3])
    return res

def RGB_to_lab(input_tensor):
    """
    convert 299*299*3 rgb image to 224*224*3 lab values, output l and a*b* channels
    """
    res = tf.image.resize(np.asarray(input_tensor),size=(224,224)) # resize image to 224*224*3 to fit in Inception-ResNet-v2
    res /= 255 # range[0,1]
    res = np.asarray(color.rgb2lab(res)) # convert rgb to lab
    l_channel = res[:, :, 0]/100  # L range [0,1]
    ab_channels = res[:, :, 1:]/127  #ab range [-1,1]
    l_channel=tf.reshape(l_channel,[-1,224,224,1])
    ab_channels=tf.reshape(ab_channels,[-1,224,224,2])
    return l_channel,ab_channels

def emb_for_fusion(input_tensor):
    """
    stack input feature representation vector 28*28 times for fusion layer 
    """
    embeddings=tf.reshape(tf.repeat(input_tensor,repeats=28*28),[-1,28,28,1536])
    return embeddings

def prep_for_encoder(input_tensor):
    """
    resize input rbg image to 224*224*3 for encoder
    """
    encoder_img=tf.image.resize(input_tensor,size=(224,224))
    return encoder_img

def load_images_list(filepath):
    '''
    Load names of train/test/val images into a list from a text file
    '''
    images_txt = open(filepath,'r')
    images_list = []
    for line in images_txt:
        images_list.append(line.strip())
    return images_list