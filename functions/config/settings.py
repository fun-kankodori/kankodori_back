"""
アプリケーション設定管理モジュール

このモジュールは観光地推薦システム全体の設定値を一元管理します。
環境変数、パス設定、モデル設定、API設定などを含みます。
"""

import os
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()


@dataclass
class PathConfig:
    """パス設定クラス"""

    # ベースディレクトリ
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # APIディレクトリ
    API_DIR: str = os.path.join(BASE_DIR, "api")
    PHOTO_DIR: str = os.path.join(API_DIR, "photo")
    QUERY_IMAGE_DIR: str = os.path.join(API_DIR, "query_image")
    QUERY_WAIT_DIR: str = os.path.join(API_DIR, "query_wait")
    ADD_DIR: str = os.path.join(API_DIR, "add")

    # データファイル
    HAKODATE_JSON: str = os.path.join(API_DIR, "hakodate_result.json")

    # 特徴量ディレクトリ
    TEXT_FEATURE_DIR: str = os.path.join(BASE_DIR, "text", "feature")
    IMAGE_FEATURE_DIR: str = os.path.join(BASE_DIR, "image", "feature")

    # 特徴量ファイル
    BERT_FEATURES_FILE: str = os.path.join(TEXT_FEATURE_DIR, "bert_features_avg.npy")
    VIT_FEATURES_FILE: str = os.path.join(IMAGE_FEATURE_DIR, "features_labels_vit.npy")


@dataclass
class ServerConfig:
    """サーバー設定クラス"""

    HOST: str = "0.0.0.0"
    PORT: int = 3110
    DEBUG: bool = True
    RELOAD: bool = True

    # CORS設定
    ALLOW_ORIGINS: List[str] = None
    ALLOW_METHODS: List[str] = None
    ALLOW_HEADERS: List[str] = None

    def __post_init__(self):
        """初期化後の処理"""
        if self.ALLOW_ORIGINS is None:
            self.ALLOW_ORIGINS = ["*"]  # 本番環境では制限すること
        if self.ALLOW_METHODS is None:
            self.ALLOW_METHODS = ["*"]
        if self.ALLOW_HEADERS is None:
            self.ALLOW_HEADERS = ["*", "ngrok-skip-browser-warning"]


@dataclass
class ModelConfig:
    """モデル設定クラス"""

    # BERT設定
    BERT_MODEL_NAME: str = "bert-base-uncased"
    BERT_TOKENIZER_NAME: str = "bert-base-uncased"

    # ViT設定
    VIT_MODEL_NAME: str = "google/vit-base-patch16-224"
    VIT_PROCESSOR_NAME: str = "google/vit-base-patch16-224"

    # 特徴量次元
    BERT_FEATURE_DIM: int = 768
    VIT_FEATURE_DIM: int = 768

    # 並列処理設定
    MAX_WORKERS: int = 20


@dataclass
class SearchConfig:
    """検索設定クラス"""

    # 検索結果の最大件数
    MAX_RESULTS: int = 20

    # ランダム画像取得数
    RANDOM_IMAGE_COUNT: int = 6

    # 類似度計算の閾値
    SIMILARITY_THRESHOLD: float = 0.1


@dataclass
class ExternalAPIConfig:
    """外部API設定クラス"""

    # Hugging Face API
    HUGGING_FACE_API_TOKEN: str = os.getenv('HUGGING_API_KEY', '')
    HUGGING_FACE_MODEL_URL: str = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo"

    # Google Translate API（必要に応じて）
    GOOGLE_TRANSLATE_API_KEY: str = os.getenv('GOOGLE_TRANSLATE_API_KEY', '')


@dataclass
class MeCabConfig:
    """MeCab設定クラス"""

    # 対象品詞
    TARGET_POS: List[str] = None

    # 最小単語長
    MIN_WORD_LENGTH: int = 2

    def __post_init__(self):
        """初期化後の処理"""
        if self.TARGET_POS is None:
            self.TARGET_POS = ["名詞", "形容詞", "動詞", "形容動詞", "形状詞"]


class Settings:
    """アプリケーション設定の統合クラス"""

    def __init__(self):
        self.paths = PathConfig()
        self.server = ServerConfig()
        self.models = ModelConfig()
        self.search = SearchConfig()
        self.external_apis = ExternalAPIConfig()
        self.mecab = MeCabConfig()

        # 必要なディレクトリの作成
        self._create_directories()

    def _create_directories(self):
        """必要なディレクトリを作成"""
        directories = [
            self.paths.API_DIR,
            self.paths.PHOTO_DIR,
            self.paths.QUERY_IMAGE_DIR,
            self.paths.QUERY_WAIT_DIR,
            self.paths.ADD_DIR,
            self.paths.TEXT_FEATURE_DIR,
            self.paths.IMAGE_FEATURE_DIR,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_cors_config(self) -> Dict:
        """CORS設定を取得"""
        return {
            "allow_origins": self.server.ALLOW_ORIGINS,
            "allow_credentials": True,
            "allow_methods": self.server.ALLOW_METHODS,
            "allow_headers": self.server.ALLOW_HEADERS,
        }


# グローバル設定インスタンス
settings = Settings()


def get_settings() -> Settings:
    """設定インスタンスを取得"""
    return settings