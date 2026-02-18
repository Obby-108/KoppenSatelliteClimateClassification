import pandas as pd
from collections import Counter

# grab data
prelim_data = pd.read_csv('/Users/abhicado/Coding Projects/grad_ml/final_project/primary_koppen_dataset.csv')
# initialize sample dict
samples_by_class = {}
for class_id in prelim_data['classification'].unique():
    # store class -> sample count
    samples_by_class[int(class_id)] = len(prelim_data[prelim_data['classification'] == class_id]['.geo'])
# convert to list for ease of use
samples_by_class = list(samples_by_class.items())
# initialize insufficiently sampled class storer
insufficient_samples_by_class = {}
for classification, sample_count in samples_by_class:
    if sample_count < 2000:
        insufficient_samples_by_class[classification] = sample_count
# initialize final (sorted) insufficiently sampled class storer
final_insufficient_points_by_class = {}
# retrieve insufficient sample counts but in a sorted manner
sorted_insufficient_points_by_class_values = sorted(list(insufficient_samples_by_class.values()))
# reverse insufficiently sampled class storer to retrieve classes by sample count
reversed_insufficient_points_by_class = dict(zip(insufficient_samples_by_class.values(), insufficient_samples_by_class.keys()))
# for each insufficient sample count, effectively sort classes by sample count
for sample_count in sorted_insufficient_points_by_class_values:
    final_insufficient_points_by_class[reversed_insufficient_points_by_class[sample_count]] = sample_count
# print final result to then use to arrange classes into pixel tiers for data generation
print(final_insufficient_points_by_class)