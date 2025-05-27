from text.text_main import similar
from image.ViT.vit_main import vit_similar

def calc(range, filename, text): #入力内容から類似度を計算
    print(filename)
    if range == 0:
        labels, features_list, sort_json = similar(text, filename)
        return labels, features_list, sort_json
    elif range == 100:
        labels, features_list, sort_json = vit_similar(text,filename)
        return labels, features_list, sort_json
    else:
        # テキストの特徴量を取得
        labels_text, features_list_text, sort_json_text = similar(text, filename)

        # 画像の特徴量を取得
        labels_image, features_list_image, sort_json_image = vit_similar(text,filename)

        # ラベルで特徴量を紐づけて調整
        label_to_features = {}

        for label, feature in zip(labels_text, features_list_text):
            label_to_features[label] = [(feature*(100-range))/100]

        for label, feature in zip(labels_image, features_list_image):
            if label in label_to_features:
                label_to_features[label].append((feature*range)/100)
            else:
                label_to_features[label] = [(feature*range)/100]
        print(f"テキスト割合: {(100-range)/100}、画像割合: {range/100}")

        # 特徴量を調整してリストにまとめる
        adjusted_features_list = [
            (label, sum(features)) for label, features in label_to_features.items()
        ]
        # 類似度の高い順にソート
        adjusted_features_list.sort(key=lambda x: x[1], reverse=True)
        
        # ソートされたラベルと特徴量を分解
        sorted_labels, sorted_features = zip(*adjusted_features_list)
        
        # ソートされたJSONを作成
        sort_json = []
        for label in sorted_labels:
            for photo in sort_json_image:
                if photo['id'] == label:
                    sort_json.append(photo)
                    break
        return list(sorted_labels), list(sorted_features), sort_json
