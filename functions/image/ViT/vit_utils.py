import os
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from concurrent.futures import ThreadPoolExecutor

# ViTモデルと特徴量抽出器のロード
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

def extract_features(img_path):
    print(img_path)
    img = Image.open(img_path).convert('RGB')
    print(img_path)
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return features.flatten()

def load_or_extract_features(image_dir, feature_dir):
    features_file = os.path.join(feature_dir, 'features_labels_vit.npy')

    if os.path.exists(features_file):
        features_labels_map = np.load(features_file, allow_pickle=True).item()
    else:
        def process_image(img_file):
            img_path = os.path.join(image_dir, img_file)
            features = extract_features(img_path)
            label = os.path.splitext(img_file)[0]  # ファイル名（拡張子抜き）をラベルとする
            return label, features

        # ディレクトリ内のすべての画像を並列で処理
        max_workers = 20  # スレッド数を設定
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, img_file) for img_file in os.listdir(image_dir) if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            features_labels_map = {}
            for future in futures:
                label, features = future.result()
                features_labels_map[label] = features

        # 特徴量とラベルを保存
        np.save(features_file, features_labels_map)

    return features_labels_map
