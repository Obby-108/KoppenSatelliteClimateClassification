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

def load_shards(filenames):
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    return dataset.map(_parse_function)
