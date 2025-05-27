import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor
import json

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
    features_list = []
    zero_vector = np.zeros(768)
    if 'title' in photo:
        title_text = photo['title']
        features = extract_features(title_text)
    else:
        features = zero_vector
    features_list.append(features)

    if 'name' in photo:
        name_text = photo['name']
        features = extract_features(name_text)
    else:
        features = zero_vector
    features_list.append(features)

    if 'tag' in photo:
        tag_text = photo['tag']
        features = extract_features(tag_text)
    else:
        features = zero_vector
    features_list.append(features)

    if 'explain' in photo:
        explain_text = photo['explain']
        features = extract_features(explain_text)
    else:
        features = zero_vector
    features_list.append(features)

    if 'caption' in photo:
        caption_text = photo['caption']
        features = extract_features(caption_text)
    else:
        features = zero_vector
    features_list.append(features)

    if 'caption_ja' in photo:
        caption_ja_text = photo['caption_ja']
        features = extract_features(caption_ja_text)
    else:
        features = zero_vector
    features_list.append(features)

    if 'location' in photo:
        location_text = photo['location']
        features = extract_features(location_text)
    else:
        features = zero_vector
    features_list.append(features)

    if 'description' in photo and '_content' in photo['description']:
        description_text = photo['description']['_content']
        features = extract_features(description_text)
    else:
        features = zero_vector
    features_list.append(features)
            
    return photo['id'], features_list


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

def load_or_extract_features(json_path, feature_dir):
    features_file = os.path.join(feature_dir, 'bert_features.npy')

    if os.path.exists(features_file):
        features_map = np.load(features_file, allow_pickle=True).item() 
    else:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        num_threads = 20
        features_map = extract_features_parallel(data, num_threads)
        np.save(features_file, features_map)

    return features_map
