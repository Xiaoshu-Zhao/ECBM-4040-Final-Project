from os.path import join, expanduser
import tensorflow as tf
from abc import abstractmethod, ABC
import multiprocessing

compression = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

class RecordWriter(tf.python_io.TFRecordWriter):

    def __init__(self, tfrecord_name, dest_folder=""):
        self.path = join(dest_folder, tfrecord_name)
        super().__init__(self.path, options=compression)

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64(single_int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[single_int]))

    @staticmethod
    def _int64_list(list_of_int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_int))

    @staticmethod
    def _float32(single_float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[single_float]))

    @staticmethod
    def _float32_list(list_of_floats):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
    

class RecordReader(ABC):

    def __init__(self, tfrecord_pattern, folder=""):
        # Normalize the folder and build the path
        tfrecord_pattern = join(expanduser(folder), tfrecord_pattern)

        # This queue will yield a filename every time it is polled
        file_matcher = tf.train.match_filenames_once(tfrecord_pattern)

        filename_queue = tf.train.string_input_producer(file_matcher)
        reader = tf.TFRecordReader(options=compression)
        tfrecord_key, self._tfrecord_serialized = reader.read(filename_queue)

        self._path = tfrecord_key
        self._read_operation = None

    @property
    def read_operation(self):
        if self._read_operation is None:
            self._read_operation = self._create_read_operation()
        return self._read_operation

    @abstractmethod
    def _create_read_operation(self):
        pass

class BatchableRecordReader(RecordReader):

    def read_batch(self, batch_size, shuffle=False):
        # Recommended configuration for these parameters (found online)
        num_threads = multiprocessing.cpu_count()
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + (num_threads + 1) * batch_size

        if shuffle:
            return tf.train.shuffle_batch(
                self.read_operation,
                batch_size,
                capacity,
                min_after_dequeue,
                num_threads,
                allow_smaller_final_batch=False,
            )
        else:
            return tf.train.batch(
                self.read_operation,
                batch_size,
                num_threads,
                capacity,
                allow_smaller_final_batch=False,
            )