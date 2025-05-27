#基本使わない: ラベルの平均求める
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

feature_dir = os.path.join(current_dir, 'feature')
features_file = os.path.join(feature_dir, 'bert_features.npy')
features_map = np.load(features_file, allow_pickle=True).item()

print(len(features_map))

features_list = []
labels = []
for photo_id, feature_list in features_map.items():
    labels.append(f"{photo_id}")
    non_zero_features = [feature for feature in feature_list if np.any(feature)]
    if non_zero_features:
        feature_avg = np.mean(non_zero_features, axis=0)
        features_list.append(feature_avg)
print(len(features_list))
print(len(labels))

new_features_map = {}
for label, feature in zip(labels, features_list):
    new_features_map[label] = feature

features_file = os.path.join(feature_dir, 'bert_features_avg.npy')
np.save(features_file, new_features_map)
