import json
import os
from bert_utils import extract_features
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def similar(data):
    # このPythonファイルのディレクトリを取得
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(current_dir, 'flickr/analy.json')
    feature_dir = os.path.join(current_dir, 'text', 'feature')

    features_file=os.path.join(feature_dir, 'bert_features.npy')
    features_map = np.load(features_file, allow_pickle=True).item() 
    #features_labels_map = load_or_extract_features(json_path, feature_dir)

    features_list = []
    labels = []
    for photo_id, feature_list in features_map.items():
        for key, features in enumerate(feature_list):
            if data:
                for photo in data:
                    if photo['id']==photo_id:
                        features_list.append(features)
                        labels.append(f"{photo_id}")
            else:
                features_list.append(features)
                labels.append(f"{photo_id}")


    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    text=input("テキストを入力してください:")
    query_features = extract_features(text)

    if np.all(query_features == 0):
        print("警告: クエリ画像の特徴ベクトルがすべて0です。")
    else:
        # コサイン類似度を計算して類似画像を特定
        cosine_similarities = cosine_similarity([query_features], features_list)
        cosine_similar_img_index = np.argmax(cosine_similarities)
        cosine_similar_img_label = labels[cosine_similar_img_index]
        print(f"コサイン類似度: {cosine_similar_img_label}")

        for photo in json_data['photo']:
            if photo['id'] == cosine_similar_img_label:
                print(f"タイトル：{photo['title']}, 場所：{photo['name']}, タグ：{photo['word']}")

                image_path = os.path.join(current_dir, 'flickr', 'photo', f"{cosine_similar_img_label}.jpg")
                if os.path.exists(image_path):
                    img = mpimg.imread(image_path)
                    plt.imshow(img)
                    plt.show()
                else:
                    print("画像が見つかりませんでした。")


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(current_dir, 'flickr/analy.json')

with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

similar(data['photo'])
