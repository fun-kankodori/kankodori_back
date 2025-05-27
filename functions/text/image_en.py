#入力画像からテキストを抽出し、日本語へ翻訳する関数
#テキスト生成モデルの変更
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, UnidentifiedImageError
import os
from argostranslate import package, translate

def trans(text):
    # argostranslateの翻訳モデルをダウンロードする
    package.update_package_index()
    available_packages = package.get_available_packages()
    package_to_install = next(
        filter(lambda x: x.from_code == 'en' and x.to_code == 'ja', available_packages)
    )
    package.install_from_path(package_to_install.download())

    # 翻訳モデルをロードする
    installed_languages = translate.get_installed_languages()
    translator = installed_languages[0].get_translation(installed_languages[1])
    translated_title = translator.translate(text)
    return translated_title

def caption_gen(filename):
    # モデルとプロセッサの読み込み
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    photo_path = os.path.join(current_dir, 'api', 'query_wait')

    image_file = f"{filename}" 
    image_path = os.path.join(photo_path, image_file)

    try:
        image = Image.open(image_path)
    except UnidentifiedImageError:
        print(f"エラー: 画像ファイルを識別できませんでした。パスを確認してください: {image_path}")
        exit(1)
    except FileNotFoundError:
        print(f"エラー: 画像ファイルが見つかりませんでした。パスを確認してください: {image_path}")
        exit(1)

    # 画像をモデルに入力できる形式に変換
    inputs = processor(images=image, return_tensors="pt")

    # キャプションの生成
    output_ids = model.generate(**inputs, max_length=16, num_beams=4)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    text=trans(caption)

    return text
