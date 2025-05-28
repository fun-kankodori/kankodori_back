"""
BERT特徴量抽出ユーティリティモジュール

このモジュールはBERTモデルを使用してテキストの特徴量抽出を行います。
観光地データの各フィールド（タイトル、名前、タグ、説明等）から特徴量を抽出し、
並列処理によって効率的に処理します。
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from transformers import BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor

from config.settings import get_settings


class BERTFeatureExtractor:
    """BERT特徴量抽出クラス"""

    def __init__(self):
        """特徴量抽出器の初期化"""
        self.settings = get_settings()

        # BERTモデルとトークナイザーの初期化
        self.tokenizer = BertTokenizer.from_pretrained(self.settings.models.BERT_TOKENIZER_NAME)
        self.model = BertModel.from_pretrained(self.settings.models.BERT_MODEL_NAME)

        # 推論モードに設定
        self.model.eval()

    def extract_features(self, text: str) -> np.ndarray:
        """
        テキストからBERT特徴量を抽出

        Args:
            text: 特徴量を抽出するテキスト

        Returns:
            np.ndarray: 抽出された特徴量ベクトル（768次元）
        """
        if not text or not text.strip():
            # 空のテキストの場合はゼロベクトルを返す
            return np.zeros(self.settings.models.BERT_FEATURE_DIM)

        try:
            # テキストをトークン化
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512  # BERTの最大長
            )

            # 勾配計算を無効化して推論実行
            with torch.no_grad():
                outputs = self.model(**inputs)

            # [CLS]トークンの特徴量を取得（文全体の表現）
            last_hidden_states = outputs.last_hidden_state
            cls_vector = last_hidden_states[0][0]  # バッチの最初の[CLS]トークン

            return cls_vector.numpy()

        except Exception as e:
            print(f"BERT特徴量抽出エラー: {e}")
            return np.zeros(self.settings.models.BERT_FEATURE_DIM)


class PhotoDataProcessor:
    """観光地写真データ処理クラス"""

    def __init__(self):
        """データプロセッサーの初期化"""
        self.feature_extractor = BERTFeatureExtractor()
        self.settings = get_settings()

    def process_photo(self, photo: Dict[str, Any]) -> Tuple[str, List[np.ndarray]]:
        """
        単一の観光地写真データから特徴量を抽出

        Args:
            photo: 観光地写真データの辞書

        Returns:
            Tuple[str, List[np.ndarray]]: (写真ID, 各フィールドの特徴量リスト)
        """
        photo_id = photo.get('id', '')
        features_list = []

        # 処理対象フィールドの定義
        text_fields = [
            'title',      # タイトル
            'name',       # 名前
            'tag',        # タグ
            'explain',    # 説明
            'caption',    # キャプション（英語）
            'caption_ja', # キャプション（日本語）
            'location'    # 場所
        ]

        # 各フィールドから特徴量を抽出
        for field in text_fields:
            text = photo.get(field, '')
            if text:
                features = self.feature_extractor.extract_features(text)
            else:
                features = np.zeros(self.settings.models.BERT_FEATURE_DIM)
            features_list.append(features)

        # description フィールドの特別処理（ネストされた構造）
        description_text = ''
        if 'description' in photo and isinstance(photo['description'], dict):
            description_text = photo['description'].get('_content', '')

        if description_text:
            features = self.feature_extractor.extract_features(description_text)
        else:
            features = np.zeros(self.settings.models.BERT_FEATURE_DIM)
        features_list.append(features)

        return photo_id, features_list

    def extract_features_parallel(
        self,
        photos_data: List[Dict[str, Any]],
        num_threads: Optional[int] = None
    ) -> Dict[str, List[np.ndarray]]:
        """
        複数の写真データから並列で特徴量を抽出

        Args:
            photos_data: 観光地写真データのリスト
            num_threads: 使用するスレッド数（Noneの場合は設定値を使用）

        Returns:
            Dict[str, List[np.ndarray]]: 写真IDと特徴量リストのマッピング
        """
        if num_threads is None:
            num_threads = self.settings.models.MAX_WORKERS

        features_map = {}

        # ThreadPoolExecutorを使用した並列処理
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 全ての写真を並列で処理
            futures = [
                executor.submit(self.process_photo, photo)
                for photo in photos_data
            ]

            # 結果を収集
            for future in futures:
                try:
                    photo_id, features_list = future.result()
                    if photo_id:  # IDが存在する場合のみ保存
                        features_map[photo_id] = features_list
                except Exception as e:
                    print(f"写真データ処理エラー: {e}")

        print(f"特徴量抽出完了: {len(features_map)}件の写真を処理")
        return features_map


class FeatureManager:
    """特徴量管理クラス"""

    def __init__(self):
        """特徴量マネージャーの初期化"""
        self.settings = get_settings()
        self.data_processor = PhotoDataProcessor()

    def load_or_extract_features(
        self,
        json_path: str,
        feature_dir: str,
        force_extract: bool = False
    ) -> Dict[str, List[np.ndarray]]:
        """
        特徴量を読み込み、または新規抽出

        Args:
            json_path: 観光地データのJSONファイルパス
            feature_dir: 特徴量保存ディレクトリ
            force_extract: 強制的に再抽出する場合True

        Returns:
            Dict[str, List[np.ndarray]]: 写真IDと特徴量リストのマッピング
        """
        # 特徴量ファイルのパス
        features_file = os.path.join(feature_dir, 'bert_features.npy')

        # 既存の特徴量ファイルをチェック
        if os.path.exists(features_file) and not force_extract:
            try:
                print(f"既存の特徴量ファイルを読み込み: {features_file}")
                features_map = np.load(features_file, allow_pickle=True).item()
                return features_map
            except Exception as e:
                print(f"特徴量ファイル読み込みエラー: {e}")
                print("新規に特徴量を抽出します")

        # 新規に特徴量を抽出
        print(f"JSONデータから特徴量を抽出: {json_path}")

        try:
            # JSONデータの読み込み
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            photos_data = data.get('photo', [])
            if not photos_data:
                print("警告: 写真データが見つかりません")
                return {}

            # 並列処理で特徴量抽出
            features_map = self.data_processor.extract_features_parallel(photos_data)

            # 特徴量を保存
            os.makedirs(feature_dir, exist_ok=True)
            np.save(features_file, features_map)
            print(f"特徴量を保存: {features_file}")

            return features_map

        except FileNotFoundError:
            print(f"JSONファイルが見つかりません: {json_path}")
            return {}
        except json.JSONDecodeError:
            print(f"JSON解析エラー: {json_path}")
            return {}
        except Exception as e:
            print(f"特徴量抽出エラー: {e}")
            return {}


# グローバル特徴量抽出器インスタンス（後方互換性のため）
_feature_extractor = None


def get_feature_extractor() -> BERTFeatureExtractor:
    """
    グローバル特徴量抽出器インスタンスを取得

    Returns:
        BERTFeatureExtractor: 特徴量抽出器インスタンス
    """
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = BERTFeatureExtractor()
    return _feature_extractor


def extract_features(text: str) -> np.ndarray:
    """
    テキストから特徴量を抽出（後方互換性のための関数）

    Args:
        text: 入力テキスト

    Returns:
        np.ndarray: 抽出された特徴量
    """
    extractor = get_feature_extractor()
    return extractor.extract_features(text)


def process_photo(photo: Dict[str, Any]) -> Tuple[str, List[np.ndarray]]:
    """
    観光地写真データを処理（後方互換性のための関数）

    Args:
        photo: 観光地写真データ

    Returns:
        Tuple[str, List[np.ndarray]]: (写真ID, 特徴量リスト)
    """
    processor = PhotoDataProcessor()
    return processor.process_photo(photo)


def extract_features_parallel(
    data: Dict[str, Any],
    num_threads: int = 4
) -> Dict[str, List[np.ndarray]]:
    """
    並列特徴量抽出（後方互換性のための関数）

    Args:
        data: 観光地データ（'photo'キーを含む辞書）
        num_threads: スレッド数

    Returns:
        Dict[str, List[np.ndarray]]: 特徴量マップ
    """
    processor = PhotoDataProcessor()
    photos_data = data.get('photo', [])
    return processor.extract_features_parallel(photos_data, num_threads)


def load_or_extract_features(json_path: str, feature_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    特徴量の読み込みまたは抽出（後方互換性のための関数）

    Args:
        json_path: JSONファイルパス
        feature_dir: 特徴量ディレクトリ

    Returns:
        Dict[str, List[np.ndarray]]: 特徴量マップ
    """
    manager = FeatureManager()
    return manager.load_or_extract_features(json_path, feature_dir)
