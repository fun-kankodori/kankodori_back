import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor
import json

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(current_dir, 'flickr/analy.json')

with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_features(text):
    """
    テキストを受け取り、BERTモデルを使用して特徴量を抽出する関数
    """
    # テキストをトークン化し、テンソルに変換
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    # モデルを推論モードに設定
    model.eval()
    # BERTモデルを通して高次元ベクトルを取得
    with torch.no_grad():
        outputs = model(**inputs)
    # 最後の隠れ層の出力を取得（[CLS]トークンのベクトルを使用）
    last_hidden_states = outputs.last_hidden_state
    cls_vector = last_hidden_states[0][0]  # [CLS]トークンのベクトル
    # ベクトルをnumpy配列に変換して返す
    return cls_vector.numpy()

def process_photo(photo):
    if 'explain' in photo:
        explain_text = photo['explain']
        features = extract_features(explain_text)
        return photo['id'], features
    return None

def extract_features_parallel(data, num_threads=4):
    features_map = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_photo, photo) for photo in data['photo']]
        for future in futures:
            result = future.result()
            if result:
                photo_id, features = result
                features_map[photo_id] = features
    return features_map

# スレッド数を指定して特徴量を抽出
num_threads = 20  # ここでスレッド数を指定
features_map = extract_features_parallel(data, num_threads)

# 特徴量を保存するファイルのパス
features_file = os.path.join(current_dir, 'flickr', 'bert_features.npy')

# 特徴量を保存
np.save(features_file, features_map)

print(f"特徴量は {features_file} に保存されました。")


