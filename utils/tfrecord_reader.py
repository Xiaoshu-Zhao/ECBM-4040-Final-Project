from os.path import join
import tensorflow as tf
from utils.prep import prep_for_encoder, emb_for_fusion

def _parse_function(record):
    """
    parse single tfrecord from the record file
    output image name, l channel, ab channel and feature vector 
    """
    features_spec = {
    "img_name": tf.io.VarLenFeature(dtype=tf.string),
    "l_channel": tf.io.FixedLenFeature([224*224], dtype=tf.float32),
    "ab_channels": tf.io.FixedLenFeature([224*224*2], dtype=tf.float32),
    "img_embedding": tf.io.FixedLenFeature([1536], dtype=tf.float32),
}
    parsed=tf.io.parse_single_example(record, features_spec)
    img=parsed['img_name']
    l=tf.reshape(parsed['l_channel'],[224,224,1])
    ab=tf.reshape(parsed['ab_channels'],[224,224,2])
    emb=tf.reshape(parsed['img_embedding'],[1536])
    return img,l,ab,emb

def batch_reader(batch_size, record_path, record_file):
    """
    read batch records from tfrecord dataset
    ouput inputs for network and ground truth
    """
    dataset=tf.data.TFRecordDataset(join(record_path, record_file))
    dataset=dataset.repeat(count=10).map(_parse_function)
    for batch in dataset.batch(batch_size):
        train_input = batch[1]
        train_embeddings = emb_for_fusion(batch[3])
        train_truth = batch[2]
        yield [train_input, train_embeddings], train_truth
        