#テキスト検索のメイン関数
import MeCab
import os
import json
import numpy as np
from text.bert_utils import extract_features
from sklearn.metrics.pairwise import cosine_similarity
from text.image_en import caption_gen

def mecab(query):
    # MeCabを使って形態素解析
    mecab = MeCab.Tagger()
    node = mecab.parseToNode(query)
    tags = set()
    while node:
        word = node.surface  # 単語を取得
        hinshi = node.feature.split(",")[0]
        if word and len(word) > 1:  # 2文字以上の単語のみを対象
            if hinshi == "名詞" or hinshi == "形容詞" or hinshi == "動詞" or hinshi == "形容動詞" or hinshi == "形状詞":
                tags.add(word)
        node = node.next
    return list(tags)

def find_photos_with_location(data, wordlist):
    matching_photos = []
    lo_wordlist = set()  # 重複を避けるためにセットを使用
    for photo in data['photo']:
        if 'location' in photo:
            location_text = photo['location']
            matched_words = [word for word in wordlist if word in location_text]
            if matched_words:
                lo_wordlist.add(location_text)  # セットに追加
                matching_photos.append(photo)
    return matching_photos, list(lo_wordlist)  # 最後にリストに変換

def similar(text,filename):
    if text == "null":
        text = caption_gen(filename)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(current_dir, 'api', 'hakodate_result.json')

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    wordlist = mecab(text)
    #print("抽出された単語:", wordlist)

    matching_photos, lo_wordlist = find_photos_with_location(data, wordlist)
    #for photo in matching_photos:
    #    print(f"ID: {photo['id']}, 場所: {photo['location']}")

    #print("地名に一致した単語:", lo_wordlist)

    feature_dir = os.path.join(current_dir, 'text', 'feature')
    features_file=os.path.join(feature_dir, 'bert_features_avg.npy')
    features_map = np.load(features_file, allow_pickle=True).item()
    #features_labels_map = load_or_extract_features(json_path, feature_dir)
    features_list = []
    labels = []
    for photo_id, feature in features_map.items():
        if matching_photos:
            for photo in matching_photos:
                if photo['id'] == photo_id:
                    features_list.append(feature)
                    labels.append(f"{photo_id}")
        else:
            features_list.append(feature)
            labels.append(f"{photo_id}")
    query_features = extract_features(text)

    if np.all(query_features == 0):
        print("警告: クエリ画像の特徴ベクトルがすべて0です。")
    else:
        # コサイン類似度を計算して類似画像を特定
        cosine_similarities = cosine_similarity([query_features], features_list)[0]  # 1次元配列に変換
        cosine_similar_img_index = np.argmax(cosine_similarities)
        cosine_similar_img_label = labels[cosine_similar_img_index]
        #print(f"コサイン類似度: {cosine_similar_img_label}、値: {cosine_similarities[cosine_similar_img_index]}")

        similarity_label_pairs = list(zip(cosine_similarities, labels))
        similarity_label_pairs.sort(key=lambda x: x[0], reverse=True)  # コサイン類似度で降順にソート

        # ソートされた結果を分解
        sorted_similarities, sorted_labels = zip(*similarity_label_pairs)
        print(text)
        print(f"テキストコサイン類似度: {len(sorted_labels)}、値: {sorted_similarities[0]}")
        sort_json = []
        existing_names = set()
        for label in sorted_labels:
            for photo in data['photo']:
                if photo['id'] == label and photo['name'] not in existing_names:
                    sort_json.append(photo)
                    existing_names.add(photo['name'])



    return sorted_labels,sorted_similarities,sort_json

