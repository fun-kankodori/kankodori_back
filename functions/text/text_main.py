"""
テキストベース検索のメインモジュール

このモジュールはテキストを使用した観光地の類似性検索を提供します。
MeCabによる形態素解析、BERTによる特徴量抽出、類似度計算を行います。
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Any, Set
from sklearn.metrics.pairwise import cosine_similarity
import MeCab

from config.settings import get_settings
from utils.file_utils import DataManager
from text.bert_utils import extract_features
from text.image_en import caption_gen


class TextAnalyzer:
    """テキスト解析クラス"""

    def __init__(self):
        """テキスト解析器の初期化"""
        self.settings = get_settings()
        self.mecab = MeCab.Tagger()

    def extract_keywords(self, text: str) -> List[str]:
        """
        MeCabを使用してテキストからキーワードを抽出

        Args:
            text: 解析対象のテキスト

        Returns:
            List[str]: 抽出されたキーワードのリスト
        """
        node = self.mecab.parseToNode(text)
        keywords = set()

        while node:
            word = node.surface
            part_of_speech = node.feature.split(",")[0]

            # 設定に基づいて単語をフィルタリング
            if (word and
                len(word) >= self.settings.mecab.MIN_WORD_LENGTH and
                part_of_speech in self.settings.mecab.TARGET_POS):
                keywords.add(word)

            node = node.next

        return list(keywords)


class LocationMatcher:
    """場所マッチングクラス"""

    def __init__(self, data_manager: DataManager):
        """
        場所マッチャーの初期化

        Args:
            data_manager: データ管理インスタンス
        """
        self.data_manager = data_manager

    def find_photos_by_location(self, keywords: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        キーワードに基づいて場所で写真を検索

        Args:
            keywords: 検索キーワードのリスト

        Returns:
            Tuple[List[Dict], List[str]]: (マッチした写真データ, マッチした場所名のリスト)
        """
        photos = self.data_manager.get_photos()
        matching_photos = []
        matched_locations = set()

        for photo in photos:
            location = photo.get('location', '')
            if not location:
                continue

            # キーワードが場所に含まれているかチェック
            matched_keywords = [keyword for keyword in keywords if keyword in location]
            if matched_keywords:
                matching_photos.append(photo)
                matched_locations.add(location)

        return matching_photos, list(matched_locations)


class TextSimilarityCalculator:
    """テキスト類似度計算クラス"""

    def __init__(self):
        """類似度計算器の初期化"""
        self.settings = get_settings()
        self.data_manager = DataManager(self.settings.paths.HAKODATE_JSON)
        self.text_analyzer = TextAnalyzer()
        self.location_matcher = LocationMatcher(self.data_manager)

    def load_features(self) -> Dict[str, np.ndarray]:
        """
        事前計算された特徴量を読み込み

        Returns:
            Dict[str, np.ndarray]: 写真IDと特徴量のマッピング
        """
        try:
            features_map = np.load(self.settings.paths.BERT_FEATURES_FILE, allow_pickle=True).item()
            return features_map
        except FileNotFoundError:
            print(f"特徴量ファイルが見つかりません: {self.settings.paths.BERT_FEATURES_FILE}")
            return {}
        except Exception as e:
            print(f"特徴量読み込みエラー: {e}")
            return {}

    def filter_features_by_photos(
        self,
        features_map: Dict[str, np.ndarray],
        target_photos: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        指定された写真に対応する特徴量をフィルタリング

        Args:
            features_map: 特徴量マップ
            target_photos: 対象の写真データリスト

        Returns:
            Tuple[List[np.ndarray], List[str]]: (特徴量リスト, ラベルリスト)
        """
        features_list = []
        labels = []

        if target_photos:
            # 指定された写真のみを対象とする
            target_ids = {photo['id'] for photo in target_photos}
            for photo_id, feature in features_map.items():
                if photo_id in target_ids:
                    features_list.append(feature)
                    labels.append(photo_id)
        else:
            # 全ての写真を対象とする
            for photo_id, feature in features_map.items():
                features_list.append(feature)
                labels.append(photo_id)

        return features_list, labels

    def calculate_similarity(
        self,
        query_features: np.ndarray,
        features_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        クエリと特徴量リストの類似度を計算

        Args:
            query_features: クエリの特徴量
            features_list: 比較対象の特徴量リスト

        Returns:
            np.ndarray: 類似度スコアの配列
        """
        if len(features_list) == 0:
            return np.array([])

        # コサイン類似度を計算
        similarities = cosine_similarity([query_features], features_list)[0]
        return similarities

    def create_sorted_results(
        self,
        similarities: np.ndarray,
        labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        類似度に基づいてソートされた結果を作成

        Args:
            similarities: 類似度スコアの配列
            labels: 対応するラベルのリスト

        Returns:
            List[Dict]: ソートされた写真データのリスト
        """
        if len(similarities) == 0 or len(labels) == 0:
            return []

        # 類似度とラベルをペアにしてソート
        similarity_label_pairs = list(zip(similarities, labels))
        similarity_label_pairs.sort(key=lambda x: x[0], reverse=True)

        # ソートされた結果から写真データを構築
        sorted_results = []
        seen_names = set()
        photos = self.data_manager.get_photos()

        for similarity, label in similarity_label_pairs:
            # 対応する写真データを検索
            for photo in photos:
                if (photo.get('id') == label and
                    photo.get('name') not in seen_names):

                    # 類似度スコアを追加
                    photo_with_score = photo.copy()
                    photo_with_score['similarity_score'] = float(similarity)

                    sorted_results.append(photo_with_score)
                    seen_names.add(photo.get('name'))
                    break

        return sorted_results


def similar(text: str, filename: str) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    """
    テキストベースの類似度検索のメイン関数

    Args:
        text: 検索テキスト
        filename: 画像ファイル名（画像キャプション生成用）

    Returns:
        Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
    """
    print(f"テキスト類似度検索開始: '{text}'")

    # nullの場合は画像キャプションを生成
    if text == "null":
        text = caption_gen(filename)
        print(f"画像キャプションを生成: '{text}'")

    # 類似度計算器の初期化
    calculator = TextSimilarityCalculator()

    try:
        # キーワード抽出
        keywords = calculator.text_analyzer.extract_keywords(text)
        print(f"抽出されたキーワード: {keywords}")

        # 場所による写真のフィルタリング
        matching_photos, matched_locations = calculator.location_matcher.find_photos_by_location(keywords)
        if matching_photos:
            print(f"場所でマッチした写真数: {len(matching_photos)}")
            print(f"マッチした場所: {matched_locations}")

        # 特徴量の読み込み
        features_map = calculator.load_features()
        if not features_map:
            print("特徴量が読み込めませんでした")
            return [], [], []

        # 対象写真の特徴量をフィルタリング
        features_list, labels = calculator.filter_features_by_photos(features_map, matching_photos)
        if not features_list:
            print("対象となる特徴量が見つかりませんでした")
            return [], [], []

        # クエリテキストの特徴量抽出
        query_features = extract_features(text)
        if np.all(query_features == 0):
            print("警告: クエリテキストの特徴量がすべて0です")
            return [], [], []

        # 類似度計算
        similarities = calculator.calculate_similarity(query_features, features_list)

        # 結果の作成
        sorted_results = calculator.create_sorted_results(similarities, labels)

        print(f"テキスト類似度計算完了: {len(sorted_results)}件, 最高スコア: {similarities.max():.4f}")

        # 結果の分解（後方互換性のため）
        sorted_similarities = [result.get('similarity_score', 0.0) for result in sorted_results]
        sorted_labels = [result.get('id', '') for result in sorted_results]

        return sorted_labels, sorted_similarities, sorted_results

    except Exception as e:
        print(f"テキスト類似度検索エラー: {e}")
        return [], [], []
