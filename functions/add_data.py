"""
観光地データ追加モジュール

このモジュールは新しい観光地データをシステムに追加する機能を提供します。
画像とテキストの特徴量抽出、データベースへの登録を行います。
"""

import os
import json
import random
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import MeCab

from config.settings import get_settings
from utils.file_utils import FileManager, DataManager
from image.ViT.vit_utils import extract_features as extract_image_features
from text.bert_utils import extract_features as extract_text_features, process_photo


class TouristSpotDataManager:
    """観光地データ管理クラス"""

    def __init__(self, mode: int = 1):
        """
        データマネージャーの初期化

        Args:
            mode: 動作モード（1: 開発モード, 2: 本番モード）
        """
        self.settings = get_settings()
        self.mode = mode
        self.data_manager = DataManager(self.settings.paths.HAKODATE_JSON)

    def generate_unique_id(self) -> str:
        """
        ユニークなIDを生成

        Returns:
            str: 生成されたユニークID
        """
        existing_photos = self.data_manager.get_photos()
        existing_ids = {photo.get('id', '') for photo in existing_photos}

        # 新しいIDを生成（既存IDと重複しないまで繰り返し）
        while True:
            new_id = str(random.randint(100000, 999999))
            if new_id not in existing_ids:
                return new_id

    def create_photo_data(
        self,
        photo_id: str,
        name: str,
        location: str,
        description: str,
        tags: str
    ) -> Dict[str, Any]:
        """
        写真データの辞書を作成

        Args:
            photo_id: 写真ID
            name: 観光地名
            location: 場所
            description: 説明
            tags: タグ（カンマ区切り）

        Returns:
            Dict[str, Any]: 写真データの辞書
        """
        return {
            "id": photo_id,
            "name": name,
            "location": location,
            "title": name,  # タイトルは名前と同じに設定
            "explain": description,
            "tag": tags,
            "caption": "",  # 英語キャプション（後で生成可能）
            "caption_ja": "",  # 日本語キャプション（後で生成可能）
            "description": {
                "_content": description
            }
        }

    def extract_features_for_new_photo(
        self,
        photo_data: Dict[str, Any],
        image_filename: str
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        新しい写真のテキストと画像特徴量を抽出

        Args:
            photo_data: 写真データ
            image_filename: 画像ファイル名

        Returns:
            tuple: (テキスト特徴量, 画像特徴量)
        """
        text_features = None
        image_features = None

        try:
            # テキスト特徴量の抽出
            _, text_features_list = process_photo(photo_data)
            if text_features_list:
                # 複数のテキストフィールドの特徴量を平均化
                text_features = np.mean(text_features_list, axis=0)

            # 画像特徴量の抽出
            image_path = os.path.join(self.settings.paths.ADD_DIR, image_filename)
            if os.path.exists(image_path):
                image_features = extract_image_features(image_path)

        except Exception as e:
            print(f"特徴量抽出エラー: {e}")

        return text_features, image_features

    def update_feature_files(
        self,
        photo_id: str,
        text_features: np.ndarray,
        image_features: np.ndarray
    ) -> bool:
        """
        特徴量ファイルを更新

        Args:
            photo_id: 写真ID
            text_features: テキスト特徴量
            image_features: 画像特徴量

        Returns:
            bool: 更新成功時True
        """
        try:
            # テキスト特徴量ファイルの更新
            if text_features is not None:
                text_features_file = self.settings.paths.BERT_FEATURES_FILE
                if os.path.exists(text_features_file):
                    text_features_map = np.load(text_features_file, allow_pickle=True).item()
                else:
                    text_features_map = {}

                text_features_map[photo_id] = text_features
                np.save(text_features_file, text_features_map)
                print(f"テキスト特徴量を更新: {photo_id}")

            # 画像特徴量ファイルの更新
            if image_features is not None:
                image_features_file = self.settings.paths.VIT_FEATURES_FILE
                if os.path.exists(image_features_file):
                    image_features_map = np.load(image_features_file, allow_pickle=True).item()
                else:
                    image_features_map = {}

                image_features_map[photo_id] = image_features
                np.save(image_features_file, image_features_map)
                print(f"画像特徴量を更新: {photo_id}")

            return True

        except Exception as e:
            print(f"特徴量ファイル更新エラー: {e}")
            return False

    def add_to_database(self, photo_data: Dict[str, Any]) -> bool:
        """
        データベースに写真データを追加

        Args:
            photo_data: 追加する写真データ

        Returns:
            bool: 追加成功時True
        """
        try:
            # 既存データの読み込み
            data = self.data_manager.load_data(force_reload=True)

            # 新しい写真データを追加
            if 'photo' not in data:
                data['photo'] = []

            data['photo'].append(photo_data)

            # データベースファイルに保存
            success = FileManager.save_json(data, self.settings.paths.HAKODATE_JSON)

            if success:
                print(f"データベースに追加: {photo_data['id']}")
                # キャッシュをクリア
                self.data_manager.clear_cache()

            return success

        except Exception as e:
            print(f"データベース追加エラー: {e}")
            return False

    def copy_image_to_photo_dir(self, image_filename: str, photo_id: str) -> bool:
        """
        画像をphotoディレクトリにコピー

        Args:
            image_filename: 元の画像ファイル名
            photo_id: 写真ID

        Returns:
            bool: コピー成功時True
        """
        try:
            import shutil

            source_path = os.path.join(self.settings.paths.ADD_DIR, image_filename)
            dest_path = os.path.join(self.settings.paths.PHOTO_DIR, f"{photo_id}.jpg")

            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                print(f"画像をコピー: {source_path} -> {dest_path}")
                return True
            else:
                print(f"元画像が見つかりません: {source_path}")
                return False

        except Exception as e:
            print(f"画像コピーエラー: {e}")
            return False


def add_data(
    image_filename: str,
    name: str,
    location: str,
    description: str,
    tags: str
) -> bool:
    """
    新しい観光地データを追加するメイン関数

    Args:
        image_filename: 画像ファイル名
        name: 観光地名
        location: 場所
        description: 説明
        tags: タグ（カンマ区切り）

    Returns:
        bool: 追加成功時True
    """
    print(f"観光地データ追加開始: {name}")

    # データマネージャーの初期化
    manager = TouristSpotDataManager(mode=1)  # 開発モード

    try:
        # ユニークIDの生成
        photo_id = manager.generate_unique_id()
        print(f"生成されたID: {photo_id}")

        # 写真データの作成
        photo_data = manager.create_photo_data(photo_id, name, location, description, tags)

        # 特徴量の抽出
        text_features, image_features = manager.extract_features_for_new_photo(photo_data, image_filename)

        # 開発モードでは実際の追加処理をスキップ
        if manager.mode == 1:
            print("開発モード: 実際のデータ追加はスキップされます")
            print(f"追加予定データ: {photo_data}")
            return True

        # 本番モードでの処理（mode == 2の場合）
        # TODO: 以下の処理を実装
        # 1. 特徴量ファイルの更新
        # success = manager.update_feature_files(photo_id, text_features, image_features)
        # if not success:
        #     return False

        # 2. データベースへの追加
        # success = manager.add_to_database(photo_data)
        # if not success:
        #     return False

        # 3. 画像ファイルのコピー
        # success = manager.copy_image_to_photo_dir(image_filename, photo_id)
        # if not success:
        #     return False

        print(f"観光地データ追加完了: {name} (ID: {photo_id})")
        return True

    except Exception as e:
        print(f"観光地データ追加エラー: {e}")
        return False
