#Mecabを用いた形態素解析を行う関数
import MeCab
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# このPythonファイルのディレクトリを取得
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(current_dir, 'flickr', 'analy.json')

with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


def mecab(photo):
    text=f"{photo['explain']},{photo['caption_ja']}"

    # MeCabを使って形態素解析
    mecab = MeCab.Tagger()
    node = mecab.parseToNode(text)

    # 名詞を抽出
    tags = set()
    while node:
        word = node.surface # 単語を取得
        hinshi = node.feature.split(",")[0]
        if hinshi == "名詞" or hinshi == "形容詞" or hinshi == "動詞" or hinshi=="形容動詞" or hinshi=="形状詞":
            tags.add(word)
        node = node.next
    # 既存のphoto['word']に新しい単語を追加
    if 'word' in photo:
        photo['word'].extend(list(tags))
    else:
        photo['word'] = list(tags)

    # 重複を削除
    photo['word'] = list(set(photo['word']))

    return photo

# スレッド数を指定
num_threads = 20

def process_photos(data):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_photo = {executor.submit(mecab, photo): photo for photo in data['photo']}
        for future in as_completed(future_to_photo):
            future.result()

process_photos(data)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
