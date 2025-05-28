"""
類似度計算メインモジュール

このモジュールは観光地推薦システムの核となる類似度計算機能を提供します。
テキストと画像の特徴量を組み合わせた検索、重み付け計算を行います。
"""

import os
from typing import List, Tuple, Dict, Any, Optional

from config.settings import get_settings
from text.text_main import search_by_text
from image.image_main import process_images


class SimilarityCalculator:
    """類似度計算クラス"""

    def __init__(self):
        """類似度計算器の初期化"""
        self.settings = get_settings()

    def calculate_text_only(
        self,
        text: str
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        テキストのみによる類似度計算

        Args:
            text: 検索テキスト

        Returns:
            Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
        """
        print(f"テキストのみで検索: '{text}'")

        try:
            return search_by_text(text)
        except Exception as e:
            print(f"テキスト検索エラー: {e}")
            return [], [], []

    def calculate_image_only(
        self,
        image_path: str,
        text: str = ""
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        画像のみによる類似度計算

        Args:
            image_path: 画像ファイルパス
            text: テキスト（画像生成用、オプション）

        Returns:
            Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
        """
        print(f"画像のみで検索: '{image_path}'")

        try:
            return process_images(image_path, text)
        except Exception as e:
            print(f"画像検索エラー: {e}")
            return [], [], []

    def calculate_combined_similarity(
        self,
        text_labels: List[str],
        text_similarities: List[float],
        image_labels: List[str],
        image_similarities: List[float],
        weight: float
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        テキストと画像の類似度を組み合わせて計算

        Args:
            text_labels: テキスト検索のラベルリスト
            text_similarities: テキスト類似度リスト
            image_labels: 画像検索のラベルリスト
            image_similarities: 画像類似度リスト
            weight: 画像の重み（0.0-1.0）

        Returns:
            Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
        """
        print(f"組み合わせ検索: テキスト重み={1-weight:.2f}, 画像重み={weight:.2f}")

        try:
            # テキストと画像の類似度を辞書に変換
            text_sim_dict = dict(zip(text_labels, text_similarities))
            image_sim_dict = dict(zip(image_labels, image_similarities))

            # 全てのユニークなラベルを取得
            all_labels = set(text_labels + image_labels)

            # 組み合わせ類似度を計算
            combined_similarities = []
            combined_labels = []

            for label in all_labels:
                text_sim = text_sim_dict.get(label, 0.0)
                image_sim = image_sim_dict.get(label, 0.0)

                # 重み付き平均を計算
                combined_sim = (1 - weight) * text_sim + weight * image_sim

                combined_similarities.append(combined_sim)
                combined_labels.append(label)

            # 類似度でソート
            sorted_pairs = sorted(
                zip(combined_similarities, combined_labels),
                key=lambda x: x[0],
                reverse=True
            )

            sorted_similarities, sorted_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

            print(f"組み合わせ検索完了: {len(sorted_labels)}件, 最高スコア: {sorted_similarities[0]:.4f}" if sorted_similarities else "組み合わせ検索完了: 0件")

            # 結果データの構築（画像検索結果を基準とする）
            # 画像検索結果にテキスト類似度情報を追加
            if image_labels and len(image_labels) > 0:
                # 画像検索結果から写真データを取得
                _, _, image_results = process_images("dummy", "")  # 実際の結果は上で計算済み

                # ソート順に並び替え
                sorted_results = []
                for label in sorted_labels:
                    for result in image_results:
                        if result.get('id') == label:
                            # 組み合わせ類似度を追加
                            result_copy = result.copy()
                            result_copy['combined_similarity'] = combined_similarities[combined_labels.index(label)]
                            result_copy['text_similarity'] = text_sim_dict.get(label, 0.0)
                            result_copy['image_similarity'] = image_sim_dict.get(label, 0.0)
                            sorted_results.append(result_copy)
                            break

                return list(sorted_labels), list(sorted_similarities), sorted_results
            else:
                return list(sorted_labels), list(sorted_similarities), []

        except Exception as e:
            print(f"組み合わせ類似度計算エラー: {e}")
            return [], [], []

    def calculate_similarity(
        self,
        range_value: int,
        image_path: str,
        text: str
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        指定された重み付けで類似度を計算

        Args:
            range_value: 重み付け値（0=テキストのみ, 100=画像のみ, 1-99=組み合わせ）
            image_path: 画像ファイルパス
            text: 検索テキスト

        Returns:
            Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
        """
        print(f"類似度計算開始: range={range_value}, image='{image_path}', text='{text}'")

        try:
            # 重み付け値の検証
            if not 0 <= range_value <= 100:
                print(f"警告: 無効な重み付け値 {range_value}。0-100の範囲で指定してください。")
                range_value = max(0, min(100, range_value))

            # テキストのみの場合
            if range_value == 0:
                return self.calculate_text_only(text)

            # 画像のみの場合
            elif range_value == 100:
                return self.calculate_image_only(image_path, text)

            # 組み合わせの場合
            else:
                # テキストと画像の両方で検索
                text_labels, text_similarities, text_results = self.calculate_text_only(text)
                image_labels, image_similarities, image_results = self.calculate_image_only(image_path, text)

                # 重みを0.0-1.0の範囲に正規化
                image_weight = range_value / 100.0

                # 組み合わせ類似度を計算
                return self.calculate_combined_similarity(
                    text_labels, text_similarities,
                    image_labels, image_similarities,
                    image_weight
                )

        except Exception as e:
            print(f"類似度計算エラー: {e}")
            return [], [], []


def calc(
    range_value: int,
    image_path: str,
    text: str
) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    """
    類似度計算のメイン関数（後方互換性のため）

    Args:
        range_value: 重み付け値（0=テキストのみ, 100=画像のみ, 1-99=組み合わせ）
        image_path: 画像ファイルパス
        text: 検索テキスト

    Returns:
        Tuple[List[str], List[float], List[Dict]]: (ラベル, 類似度, ソート済み結果)
    """
    calculator = SimilarityCalculator()
    return calculator.calculate_similarity(range_value, image_path, text)
