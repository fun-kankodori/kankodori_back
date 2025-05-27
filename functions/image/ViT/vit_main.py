import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from image.ViT.vit_utils import extract_features
import json
from image.ai_image import generate_image

def vit_similar(text,image_path):
    if image_path == "null":
        image_path = generate_image(text)

    # このPythonファイルのディレクトリを取得
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 画像ディレクトリのパス
    image_dir = os.path.join(current_dir, 'api/photo')
    feature_dir = os.path.join(current_dir, 'image','feature')
    json_path = os.path.join(current_dir, 'api', 'hakodate_result.json')
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 特徴量とラベルのリストを取得
    features_file = os.path.join(feature_dir, 'features_labels_vit.npy')
    features_map = np.load(features_file, allow_pickle=True).item()

    features_list = []
    labels = []
    for photo_id, feature in features_map.items():
        features_list.append(feature)
        labels.append(f"{photo_id}")

    # features_listを2次元配列に変換
    features_list = np.array(features_list)
    if features_list.ndim == 1:
        features_list = features_list.reshape(1, -1)

    query_img_path = os.path.join(current_dir, 'api/query_wait', image_path)  # クエリ画像のパスを指定
    print(f"クエリ画像のパス: {query_img_path}")
    query_features = extract_features(query_img_path)
    if np.all(query_features == 0):
        print("警告: クエリ画像の特徴ベクトルがすべて0です。")
    else:
        # クエリ特徴量を2次元配列に変換
        query_features = query_features.reshape(1, -1)
        # コサイン類似度を計算して類似画像を特定
        cosine_similarities = cosine_similarity(features_list, query_features).flatten()
        cosine_similar_img_index = np.argmax(cosine_similarities)
        cosine_similar_img_label = labels[cosine_similar_img_index]
        #print(f"コサイン類似度で最も類似している画像: {cosine_similar_img_label}")
        #print(f"類似度: {cosine_similarities.shape}、ラベル: {len(labels)}")
        similarity_label_pairs = list(zip(cosine_similarities, labels))
        similarity_label_pairs.sort(key=lambda x: x[0], reverse=True)  # コサイン類似度で降順にソート

        # ソートされた結果を分解
        sorted_similarities, sorted_labels = zip(*similarity_label_pairs)
        print(f"画像コサイン類似度: {len(sorted_labels)}、値: {sorted_similarities[0]}")
        sort_json = []
        existing_names = set()
        for label in sorted_labels:
            for photo in data['photo']:
                if photo['id'] == label and photo['name'] not in existing_names:
                    sort_json.append(photo)
                    existing_names.add(photo['name'])
    '''
    # クエリ画像を削除
    if os.path.exists(query_img_path):
        os.remove(query_img_path)
        print(f"クエリ画像 {query_img_path} を削除しました。")
    else:
        print(f"クエリ画像 {query_img_path} が見つかりませんでした。")
    '''
    query_image_path = os.path.join(current_dir, 'api/query_image', image_path)
    if os.path.exists(query_image_path):
        try:
            os.remove(query_img_path)
            print(f"{query_img_path} を削除しました。")
        except Exception as e:
            print(f"{query_img_path} の削除中にエラーが発生しました: {e}")
    return sorted_labels,sorted_similarities,sort_json
