import json
from PIL import Image
import os
from image.ViT.vit_main import vit_similar

def process_images(image_path: str) -> list:
    api_path=os.path.join(os.path.dirname(__file__), "api")
    json_file_path = os.path.join(api_path, "hakodate_result.json")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    processed_image_paths = vit_similar(os.path.basename(image_path))

    matched_paths = []
    for i in range(len(processed_image_paths)):
        processed_image_paths[i] = list(processed_image_paths[i])  # タプルをリストに変換

        # JSONのid要素と一致するものを適切な配列に格納
        for pthoto in json_data['photo']:
            if pthoto['id'] == processed_image_paths[i][0]:
                matched_paths.append(pthoto)

        processed_image_paths[i][0] = f"{processed_image_paths[i][0]}.jpg"
        print(matched_paths)

    return processed_image_paths
