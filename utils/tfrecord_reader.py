from os.path import join
import tensorflow as tf
from utils.prep import prep_for_encoder, emb_for_fusion

def _parse_function(record):
    features_spec = {
    "img_name": tf.io.VarLenFeature(dtype=tf.string),
    "l_channel": tf.io.FixedLenFeature([299*299], dtype=tf.float32),
    "ab_channels": tf.io.FixedLenFeature([299*299*2], dtype=tf.float32),
    "img_embedding": tf.io.FixedLenFeature([1536], dtype=tf.float32),
}
    parsed=tf.io.parse_single_example(record, features_spec)
    img=parsed['img_name']
    l=tf.reshape(parsed['l_channel'],[299,299,1])
    ab=tf.reshape(parsed['ab_channels'],[299,299,2])
    emb=tf.reshape(parsed['img_embedding'],[1536])
    return img,l,ab,emb

def batch_reader(batch_size, record_path, record_file):
    dataset=tf.data.TFRecordDataset(join(record_path, record_file))
    dataset=dataset.shuffle(buffer_size=2*batch_size).map(_parse_function)
    for batch in dataset.repeat(10).batch(batch_size):
        train_input = prep_for_encoder(batch[1])
        train_embeddings = emb_for_fusion(batch[3])
        train_truth = prep_for_encoder(batch[2])
        yield [train_input, train_embeddings], train_truth
        