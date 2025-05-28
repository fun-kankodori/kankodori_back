"""
画像処理メインモジュール

このモジュールは画像を使用した観光地の類似性検索処理を提供します。
Vision Transformerを使用した画像特徴量抽出と類似性計算を行います。
"""

import json
import os
from typing import List, Dict, Any
from PIL import Image

from image.ViT.vit_main import vit_similar


class ImageProcessor:
    """
    画像処理と類似性検索を行うクラス
    """

    def __init__(self):
        """
        画像処理器の初期化
        JSONデータファイルのパスを設定
        """
        self.api_path = os.path.join(os.path.dirname(__file__), "api")
        self.json_file_path = os.path.join(self.api_path, "hakodate_result.json")

    def load_photo_data(self) -> List[Dict[str, Any]]:
        """
        観光地の写真データをJSONファイルから読み込む

        Returns:
            List[Dict]: 観光地の写真データリスト

        Raises:
            FileNotFoundError: JSONファイルが見つからない場合
            json.JSONDecodeError: JSONの解析に失敗した場合
        """
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            return json_data.get('photo', [])
        except FileNotFoundError:
            print(f"エラー: JSONファイルが見つかりません: {self.json_file_path}")
            raise
        except json.JSONDecodeError:
            print(f"エラー: JSONファイルの解析に失敗しました: {self.json_file_path}")
            raise

    def find_matching_photos(self, similarity_results: List[tuple], photo_data: List[Dict]) -> List[Dict]:
        """
        類似性検索結果に対応する観光地写真データを検索

        Args:
            similarity_results: ViTから得られた類似性結果のリスト（IDと類似度のタプル）
            photo_data: 観光地写真データのリスト

        Returns:
            List[Dict]: マッチした観光地データのリスト
        """
        matched_photos = []

        for i, (photo_id, similarity_score) in enumerate(similarity_results):
            # 対応する観光地データを検索
            for photo in photo_data:
                if photo.get('id') == photo_id:
                    # 類似度スコアも含めた情報を追加
                    photo_with_score = photo.copy()
                    photo_with_score['similarity_score'] = similarity_score
                    photo_with_score['rank'] = i + 1
                    matched_photos.append(photo_with_score)
                    break
            else:
                print(f"警告: ID '{photo_id}' に対応する観光地データが見つかりません")

        return matched_photos


def process_images(image_path: str) -> List[List]:
    """
    画像を処理して類似した観光地を検索する

    Args:
        image_path: 処理対象の画像ファイルパス

    Returns:
        List[List]: 処理された画像パスと類似度のリスト

    Note:
        この関数は後方互換性のために残されています。
        新しいコードではImageProcessorクラスの使用を推奨します。
    """
    print(f"画像処理を開始します: {image_path}")

    try:
        # 画像処理器のインスタンス化
        processor = ImageProcessor()

        # 観光地データの読み込み
        photo_data = processor.load_photo_data()

        # ViTを使用した類似性検索
        image_filename = os.path.basename(image_path)
        similarity_results = vit_similar(image_filename)

        # 結果をリスト形式に変換（後方互換性のため）
        processed_results = []
        for i, (photo_id, similarity_score) in enumerate(similarity_results):
            # [photo_id, similarity_score] の形式でリストに変換
            result_item = [photo_id, similarity_score]
            processed_results.append(result_item)

            # ファイル拡張子を追加（元のコードとの互換性）
            result_item[0] = f"{result_item[0]}.jpg"

        # マッチした観光地データを取得
        matched_photos = processor.find_matching_photos(similarity_results, photo_data)

        print(f"処理完了: {len(matched_photos)}件の類似観光地が見つかりました")
        print(f"マッチした観光地: {matched_photos}")

        return processed_results

    except Exception as e:
        print(f"エラー: 画像処理中に問題が発生しました: {e}")
        raise
