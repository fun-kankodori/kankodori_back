"""
ファイル操作ユーティリティモジュール

このモジュールはファイルやディレクトリの操作に関する共通関数を提供します。
JSON読み込み、画像ファイル処理、ディレクトリ操作などを含みます。
"""

import os
import json
import shutil
import random
from typing import List, Dict, Any, Optional
from PIL import Image


class FileManager:
    """ファイル管理クラス"""

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """
        JSONファイルを読み込む

        Args:
            file_path: JSONファイルのパス

        Returns:
            Dict: JSONデータ

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            json.JSONDecodeError: JSON解析エラーの場合
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSONファイルが見つかりません: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON解析エラー: {file_path}", e.doc, e.pos)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> bool:
        """
        JSONファイルに保存

        Args:
            data: 保存するデータ
            file_path: 保存先ファイルパス

        Returns:
            bool: 保存成功時True
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"JSON保存エラー: {e}")
            return False

    @staticmethod
    def save_uploaded_file(upload_file, save_path: str) -> bool:
        """
        アップロードされたファイルを保存

        Args:
            upload_file: FastAPIのUploadFileオブジェクト
            save_path: 保存先パス

        Returns:
            bool: 保存成功時True
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            return True
        except Exception as e:
            print(f"ファイル保存エラー: {e}")
            return False

    @staticmethod
    def get_image_files(directory: str) -> List[str]:
        """
        ディレクトリから画像ファイルリストを取得

        Args:
            directory: 検索対象ディレクトリ

        Returns:
            List[str]: 画像ファイル名のリスト
        """
        if not os.path.exists(directory):
            return []

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_files = []

        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)

        return image_files

    @staticmethod
    def get_random_images(directory: str, count: int = 6) -> List[str]:
        """
        ディレクトリからランダムに画像を選択

        Args:
            directory: 画像ディレクトリ
            count: 選択する画像数

        Returns:
            List[str]: ランダムに選択された画像ファイル名のリスト
        """
        image_files = FileManager.get_image_files(directory)
        if not image_files:
            return []

        random_count = min(count, len(image_files))
        return random.sample(image_files, random_count)

    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """
        画像ファイルの妥当性を検証

        Args:
            file_path: 画像ファイルパス

        Returns:
            bool: 有効な画像ファイルの場合True
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    @staticmethod
    def clean_temp_files(directory: str, max_age_minutes: int = 60) -> int:
        """
        一時ファイルをクリーンアップ

        Args:
            directory: クリーンアップ対象ディレクトリ
            max_age_minutes: 削除対象の経過時間（分）

        Returns:
            int: 削除したファイル数
        """
        import time

        if not os.path.exists(directory):
            return 0

        deleted_count = 0
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"一時ファイルを削除: {file_path}")
                    except Exception as e:
                        print(f"ファイル削除エラー: {e}")

        return deleted_count

    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """
        ディレクトリの存在を確認し、なければ作成

        Args:
            directory: ディレクトリパス

        Returns:
            bool: 成功時True
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            print(f"ディレクトリ作成エラー: {e}")
            return False


class DataManager:
    """データ管理クラス"""

    def __init__(self, json_path: str):
        """
        データマネージャーの初期化

        Args:
            json_path: メインデータファイルのパス
        """
        self.json_path = json_path
        self._data_cache: Optional[Dict[str, Any]] = None

    def load_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        データを読み込み（キャッシュ機能付き）

        Args:
            force_reload: 強制的に再読み込みする場合True

        Returns:
            Dict: 読み込まれたデータ
        """
        if self._data_cache is None or force_reload:
            self._data_cache = FileManager.load_json(self.json_path)

        return self._data_cache

    def get_photos(self) -> List[Dict[str, Any]]:
        """
        写真データのリストを取得

        Returns:
            List[Dict]: 写真データのリスト
        """
        data = self.load_data()
        return data.get('photo', [])

    def find_photo_by_id(self, photo_id: str) -> Optional[Dict[str, Any]]:
        """
        IDで写真データを検索

        Args:
            photo_id: 写真ID

        Returns:
            Optional[Dict]: 見つかった写真データ、なければNone
        """
        photos = self.get_photos()
        for photo in photos:
            if photo.get('id') == photo_id:
                return photo
        return None

    def filter_photos_by_location(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        場所のキーワードで写真をフィルタリング

        Args:
            keywords: 検索キーワードのリスト

        Returns:
            List[Dict]: マッチした写真データのリスト
        """
        photos = self.get_photos()
        matching_photos = []

        for photo in photos:
            location = photo.get('location', '')
            if any(keyword in location for keyword in keywords):
                matching_photos.append(photo)

        return matching_photos

    def clear_cache(self):
        """データキャッシュをクリア"""
        self._data_cache = None