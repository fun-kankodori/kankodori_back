#テキストから画像を生成する関数. おそらく欄時間に32回のリクエスト可能
import os
import requests
from dotenv import load_dotenv
from googletrans import Translator

# .envファイルから環境変数をロード
load_dotenv()

# 環境変数からAPIキーを取得
HF_API_TOKEN = os.getenv('HUGGING_API_KEY')

def generate_image(prompt):
    # プロンプトを英語に翻訳
    translator = Translator()
    translated_prompt = translator.translate(prompt, src='ja', dest='en').text

    # Hugging Face Inference APIを呼び出して画像を生成
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo",
        headers=headers,
        json={"inputs": translated_prompt}
    )

    if response.status_code == 200:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(current_dir, 'api/query_wait', prompt)
        # 画像を保存
        with open(f"{file_path}.jpg", "wb") as f:
            f.write(response.content)
        return f"{prompt}.jpg"  # ファイル名のみを返す
    else:
        raise Exception(f"Failed to generate image: {response.status_code}, {response.text}")
