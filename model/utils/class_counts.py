import multiprocessing
import tensorflow as tf
import numpy as np

def _label_parse_function(example_proto):
    feature_description = {
        'classification': tf.io.FixedLenFeature([1], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    return tf.cast(parsed['classification'][0], tf.int64) - 1

def get_class_counts(file_list, num_classes=30):
    cores = multiprocessing.cpu_count()

    # Build dataset for iteration
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=cores,
        block_length=64,
        num_parallel_calls=cores,
        deterministic=False
    )
    dataset = dataset.map(_label_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(2048).prefetch(tf.data.AUTOTUNE)

    counts = np.zeros(num_classes, dtype=np.int64)

    for batch in dataset:
        labels = batch.numpy()
        unique, batch_counts = np.unique(labels, return_counts=True)
        counts[unique] += batch_counts

    return counts
