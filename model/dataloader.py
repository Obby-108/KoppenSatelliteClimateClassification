import multiprocessing

import tensorflow as tf

def _parse_function(example_proto):
    # Define the specific keys found in the shards
    band_keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    feature_description = {
        'classification': tf.io.FixedLenFeature([1], tf.float32),
    }

    for key in band_keys:
        feature_description[key] = tf.io.FixedLenFeature([129 * 129], tf.float32)

    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Reconstruct the 12-channel tensor
    bands = []
    for key in band_keys:
        band = tf.reshape(parsed_features[key], [129, 129, 1])
        bands.append(band)

    # Stack along the last axis to get (129, 129, 12)
    full_tensor = tf.concat(bands, axis=-1)

    # Get the label
    label = tf.cast(parsed_features['classification'][0], tf.int64)

    return full_tensor, label


def load_shards(filenames, batch_size=512, stats=None, is_training=True):
    cores = multiprocessing.cpu_count()
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=cores,
        block_length=16,
        num_parallel_calls=cores,
        deterministic=False
    )

    dataset = dataset.map(_parse_function, num_parallel_calls=cores)

    if stats is not None:
        m_list, s_list = stats

        # We reshape to (1, 1, 12) so they broadcast correctly over (129, 129, 12)
        means_tensor = tf.constant(m_list, dtype=tf.float32)
        means_tensor = tf.reshape(means_tensor, [1, 1, 12])

        stds_tensor = tf.constant(s_list, dtype=tf.float32)
        stds_tensor = tf.reshape(stds_tensor, [1, 1, 12])

        def normalize_fn(image, label):
            image = tf.cast(image, tf.float32)
            image = (image - means_tensor) / (stds_tensor + 1e-7)
            return image, label

        dataset = dataset.map(normalize_fn, num_parallel_calls=cores)

    # Shuffle for training data
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
