import numpy as np

# Function to calculate mean and std of training and testing sets
def calculate_stats(tf_dataset, num_samples=5000, clip_val=4000):
    images_iterator = tf_dataset.unbatch().take(num_samples).as_numpy_iterator()
    all_images = np.stack([img for img, _ in images_iterator])

    # Clip outliers (clouds/snow/sensor errors) BEFORE calculating stats
    all_images = np.clip(all_images, 0, clip_val)

    means = np.mean(all_images, axis=(0, 1, 2))
    stds = np.std(all_images, axis=(0, 1, 2))

    return means.tolist(), stds.tolist()
