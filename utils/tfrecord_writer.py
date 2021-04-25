from os import listdir, makedirs
from os.path import join, isdir
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import Model
from utils.prep import prep_for_inception, RGB_to_lab

"""
initialize Inception-ResNet-v2 Feature Extractor
"""
pre_trained_model = InceptionResNetV2(weights='imagenet', input_shape=(299,299,3))
feature_extractor = Model(inputs=pre_trained_model.input,
                              outputs=pre_trained_model.layers[-2].output)
    
def get_emb(img):
    """
    output 1536*1 feature representation of image
    """
    incep_img=prep_for_inception(img)
    embs=feature_extractor.predict(incep_img)
    return embs
    
def tfrecordwriter(resized_dir, img_list, record_path, file_name):
    """
    write tfrecord to destination file
    """
    if not isdir(resized_dir):
        raise Exception('No resized images found')
    if not isdir(record_path):
        makedirs(record_path)
        
    writer = tf.io.TFRecordWriter(join(record_path,file_name))    
    count=0
    for file in open(img_list):
        file=file.rstrip('\n')
        img_path = join(resized_dir,file)
        img_name = file.encode('utf_8')
        img=Image.open(img_path)
        embs = get_emb(img)
        l_channel, ab_channels = RGB_to_lab(img)
            
        example = tf.train.Example(features=tf.train.Features(
            feature={'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
                     'l_channel':tf.train.Feature(float_list=tf.train.FloatList(value=l_channel.numpy().flatten())),
                     'ab_channels':tf.train.Feature(float_list=tf.train.FloatList(value=ab_channels.numpy().flatten())),
                     'img_embedding':tf.train.Feature(float_list=tf.train.FloatList(value=embs.flatten()))}))
        writer.write(example.SerializeToString())
        count+=1
        if count%10000==0:
            print(f'{count} records wrote')
    writer.close()
    print(f'Successfully write {count} records, stored at {join(record_path, file_name)}')