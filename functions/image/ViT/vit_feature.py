# VGG16モデルを使用して画像の特徴量を抽出する
import os
from vit_utils import load_or_extract_features

# このPythonファイルのディレクトリを取得
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

image_dir = os.path.join(current_dir, 'flickr', 'photo')
feature_dir = os.path.join(current_dir, 'image', 'feature')
os.makedirs(feature_dir, exist_ok=True)

# 特徴量とラベルをロードまたは抽出
features_labels_map = load_or_extract_features(image_dir, feature_dir)
