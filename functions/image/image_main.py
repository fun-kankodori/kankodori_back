"""
画像ベース検索のメインモジュール

このモジュールは画像を使用した観光地の類似性検索を提供します。
Vision Transformerによる特徴量抽出、類似度計算を行います。
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_settings
from utils.file_utils import DataManager
from image.ViT.vit_utils import extract_features
from image.ai_image import generate_image


class ImageProcessor:
    """画像処理クラス"""

    def __init__(self):
        """画像プロセッサーの初期化"""
        self.settings = get_settings()
        self.data_manager = DataManager(self.settings.paths.HAKODATE_JSON)

    def load_features(self) -> Dict[str, np.ndarray]:
        """
        事前計算された画像特徴量を読み込み

        Returns:
            Dict[str, np.ndarray]: 写真IDと特徴量のマッピング
        """
        try:
            features_map = np.load(self.settings.paths.VIT_FEATURES_FILE, allow_pickle=True).item()
            return features_map
        except FileNotFoundError:
            print(f"画像特徴量ファイルが見つかりません: {self.settings.paths.VIT_FEATURES_FILE}")
            return {}
        except Exception as e:
            print(f"画像特徴量読み込みエラー: {e}")
            return {}

    def process_query_image(self, image_path: str, text: str = "") -> str:
        """
        クエリ画像を処理し、必要に応じて生成

        Args:
            image_path: 画像ファイルパス
            text: テキスト（画像生成用）

        Returns:
            str: 処理済み画像のパス
        """
        if image_path == "null" and text:
            # テキストから画像を生成
            generated_path = generate_image(text)
            return generated_path

        return image_path

    def match_photos_with_data(
        self,
        sorted_labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        ソートされたラベルと写真データをマッチング

        Args:
            sorted_labels: ソートされた写真IDのリスト

        Returns:
            List[Dict]: マッチした写真データのリスト
        """
        photos = self.data_manager.get_photos()
        matched_photos = []
        seen_names = set()

        for label in sorted_labels:
            for photo in photos:
                if (photo.get('id') == label and
                    photo.get('name') not in seen_names):

                    matched_photos.append(photo)
                    seen_names.add(photo.get('name'))
                    break

        return matched_photos

    def calculate_similarity(
        self,
        query_image_path: str
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        画像類似度を計算

        Args:
            query_image_path: クエリ画像のパス

        Returns:
            Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
        """
        try:
            # 特徴量の読み込み
            features_map = self.load_features()
            if not features_map:
                print("画像特徴量が読み込めませんでした")
                return [], [], []

            # 特徴量とラベルのリストを作成
            features_list = []
            labels = []
            for photo_id, feature in features_map.items():
                features_list.append(feature)
                labels.append(photo_id)

            # 特徴量を2次元配列に変換
            features_array = np.array(features_list)
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)

            # クエリ画像の完全パス
            full_query_path = os.path.join(self.settings.paths.QUERY_WAIT_DIR, query_image_path)
            print(f"クエリ画像のパス: {full_query_path}")

            # クエリ画像の特徴量抽出
            query_features = extract_features(full_query_path)
            if np.all(query_features == 0):
                print("警告: クエリ画像の特徴量がすべて0です")
                return [], [], []

            # クエリ特徴量を2次元配列に変換
            query_features = query_features.reshape(1, -1)

            # コサイン類似度を計算
            similarities = cosine_similarity(features_array, query_features).flatten()

            # 類似度とラベルをペアにしてソート
            similarity_label_pairs = list(zip(similarities, labels))
            similarity_label_pairs.sort(key=lambda x: x[0], reverse=True)

            # ソートされた結果を分解
            sorted_similarities, sorted_labels = zip(*similarity_label_pairs)

            print(f"画像類似度計算完了: {len(sorted_labels)}件, 最高スコア: {sorted_similarities[0]:.4f}")

            # 写真データとマッチング
            matched_photos = self.match_photos_with_data(sorted_labels)

            # 類似度スコアを追加
            for i, photo in enumerate(matched_photos):
                if i < len(sorted_similarities):
                    photo['similarity_score'] = float(sorted_similarities[i])

            # クエリ画像のクリーンアップ
            self._cleanup_query_image(full_query_path, query_image_path)

            return list(sorted_labels), list(sorted_similarities), matched_photos

        except Exception as e:
            print(f"画像類似度計算エラー: {e}")
            return [], [], []

    def _cleanup_query_image(self, full_query_path: str, query_image_path: str):
        """
        クエリ画像をクリーンアップ

        Args:
            full_query_path: クエリ画像の完全パス
            query_image_path: クエリ画像のファイル名
        """
        # query_imageディレクトリにも同じファイルがあるかチェック
        query_image_dir_path = os.path.join(self.settings.paths.QUERY_IMAGE_DIR, query_image_path)

        if os.path.exists(query_image_dir_path):
            try:
                os.remove(full_query_path)
                print(f"クエリ画像を削除: {full_query_path}")
            except Exception as e:
                print(f"クエリ画像削除エラー: {e}")


def process_images(image_path: str, text: str = "") -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    """
    画像ベースの類似度検索のメイン関数

    Args:
        image_path: 画像ファイルパス
        text: テキスト（画像生成用、オプション）

    Returns:
        Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
    """
    print(f"画像類似度検索開始: '{image_path}'")

    processor = ImageProcessor()

    try:
        # クエリ画像の処理
        processed_image_path = processor.process_query_image(image_path, text)

        # 類似度計算
        return processor.calculate_similarity(processed_image_path)

    except Exception as e:
        print(f"画像処理エラー: {e}")
        return [], [], []