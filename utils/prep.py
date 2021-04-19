import time
import tensorflow as tf
import skimage.color as color
import numpy as np

def prep_for_inception(input_tensor):
    res = tf.image.rgb_to_grayscale(np.asarray(input_tensor))
    res = tf.image.grayscale_to_rgb(res)
    res = 2 * res / 255 - 1 # range[-1,1]
    res = tf.cast(tf.reshape(res, [-1, 299, 299, 3]),dtype=tf.float32)
    return res

def RGB_to_lab(input_tensor):
    res = np.asarray(color.rgb2lab(input_tensor))
    l_channel = 2 * res[:, :, 0] / 100 - 1 # L range [-1,1]
    ab_channels = res[:, :, 1:] / 127 #ab range [-1,1]
    l_channel=tf.cast(tf.reshape(l_channel,[-1,299,299,1]),dtype=tf.float32)
    ab_channels=tf.cast(tf.reshape(ab_channels,[-1,299,299,2]),dtype=tf.float32)
    return l_channel,ab_channels

def emb_for_fusion(input_tensor):
    embeddings=tf.reshape(tf.repeat(input_tensor,repeats=28*28),[-1,28,28,1536])
    return embeddings

def prep_for_encoder(input_tensor):
    encoder_img=tf.image.resize(input_tensor,size=(224,224))
    return encoder_img