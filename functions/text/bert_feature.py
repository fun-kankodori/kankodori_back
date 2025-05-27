import os
from bert_utils import load_or_extract_features

# このPythonファイルのディレクトリを取得
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

json_path = os.path.join(current_dir, 'flickr/analy.json')
feature_dir = os.path.join(current_dir, 'text', 'feature')
os.makedirs(feature_dir, exist_ok=True)

# 特徴量とラベルをロードまたは抽出
features_labels_map = load_or_extract_features(json_path, feature_dir)

# 特徴量とラベルをリストに変換
features_list = list(features_labels_map.values())
labels = list(features_labels_map.keys())

print("特徴量の数:", len(features_list[5]))
print("ラベルの数:", len(labels))
