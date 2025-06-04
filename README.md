# 観光地推薦システム (Kankodori Backend)

## 概要

観光地推薦システムは、テキストと画像の両方を使用して観光地の類似性検索を行う AI ベースのシステムです。ユーザーが入力したテキストや画像に基づいて、最適な観光地を推薦します。

## 技術スタック

- **フレームワーク**: FastAPI
- **機械学習**:
  - BERT (テキスト特徴量抽出)
  - Vision Transformer (画像特徴量抽出)
- **自然言語処理**: MeCab (形態素解析)
- **画像処理**: PIL, OpenCV
- **データベース**: JSON 形式のファイルベース
- **デプロイ**: Docker, Heroku 対応

## システム構成

### 新しいアーキテクチャ（リファクタリング後）

```
functions/
├── config/                    # 設定管理
│   └── settings.py           # アプリケーション設定
├── utils/                     # 共通ユーティリティ
│   └── file_utils.py         # ファイル操作ユーティリティ
├── text/                      # テキスト処理モジュール
│   ├── text_main.py          # テキスト検索メイン
│   ├── bert_utils.py         # BERT特徴量抽出
│   └── image_en.py           # 画像キャプション生成
├── image/                     # 画像処理モジュール
│   ├── image_main.py         # 画像検索メイン
│   ├── ai_image.py           # AI画像生成
│   └── ViT/                  # Vision Transformer
│       └── vit_utils.py      # ViT特徴量抽出
├── main.py                    # FastAPI メインアプリケーション
├── result.py                  # 類似度計算エンジン
├── add_data.py               # データ追加機能
└── api/                      # データディレクトリ
    ├── photo/                # 観光地画像
    ├── query_image/          # クエリ用サンプル画像
    ├── query_wait/           # アップロード一時保存
    ├── add/                  # 新規追加画像
    ├── features/             # 特徴量ファイル
    └── hakodate_result.json  # 観光地データベース
```

### 主要コンポーネント

#### 1. 設定管理 (`config/settings.py`)

- アプリケーション全体の設定を一元管理
- パス設定、モデル設定、API 設定、検索設定
- 環境変数サポート

#### 2. ユーティリティ (`utils/file_utils.py`)

- ファイル操作の共通機能
- JSON 読み込み、画像処理、ディレクトリ管理
- データマネージャークラス

#### 3. テキスト処理 (`text/`)

- **text_main.py**: テキストベース検索のメイン処理
- **bert_utils.py**: BERT 特徴量抽出（並列処理対応）
- MeCab 形態素解析、類似度計算

#### 4. 画像処理 (`image/`)

- **image_main.py**: 画像ベース検索のメイン処理
- **ViT/vit_utils.py**: Vision Transformer 特徴量抽出
- **ai_image.py**: AI 画像生成機能

#### 5. 類似度計算エンジン (`result.py`)

- テキストと画像の重み付け組み合わせ
- 3 つの検索モード（テキストのみ、画像のみ、組み合わせ）
- 高度な類似度計算アルゴリズム

## API 仕様

### エンドポイント

#### 1. ヘルスチェック

```
GET /
```

**レスポンス例:**

```json
{
  "status": "healthy",
  "message": "観光地推薦システムAPIが正常に動作しています"
}
```

#### 2. 観光地検索

```
POST /upload
```

**パラメータ:**

- `file`: 画像ファイル (multipart/form-data)
- `text`: 検索テキスト (string)
- `range`: 重み付け値 (integer, 0-100)
  - 0: テキストのみ
  - 100: 画像のみ
  - 1-99: テキストと画像の組み合わせ

**レスポンス例:**

```json
{
  "range": 50,
  "result": [
    {
      "id": "123456",
      "name": "函館山",
      "location": "北海道函館市",
      "explain": "函館の象徴的な観光地",
      "tag": "夜景,山,観光",
      "image_url": "http://localhost:3110/api/photo/123456.jpg",
      "similarity_score": 0.95,
      "combined_similarity": 0.92,
      "text_similarity": 0.89,
      "image_similarity": 0.95
    }
  ],
  "total_found": 150
}
```

#### 3. ランダム画像取得

```
GET /api/query_image
```

**レスポンス例:**

```json
{
  "images": ["sample1.jpg", "sample2.jpg", "sample3.jpg"]
}
```

#### 4. サンプル画像取得

```
GET /api/query_image/{image_name}
```

#### 5. 新規観光地追加

```
POST /add_place
```

**パラメータ:**

- `file`: 画像ファイル
- `name`: 観光地名
- `location`: 場所
- `description`: 説明
- `tags`: タグ（カンマ区切り）

#### 6. システム統計

```
GET /api/stats
```

**レスポンス例:**

```json
{
  "total_photos": 1500,
  "temp_files_cleaned": 5,
  "system_status": "healthy"
}
```

## セットアップ手順

### 1. 環境構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd kankodori/backend/functions

# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. 環境変数設定

`.env`ファイルを作成:

```env
# サーバー設定
SERVER_HOST=0.0.0.0
SERVER_PORT=3110
DEBUG=True

# CORS設定
CORS_ORIGINS=["*"]

# モデル設定
BERT_MODEL_NAME=bert-base-uncased
BERT_TOKENIZER_NAME=bert-base-uncased
VIT_MODEL_NAME=google/vit-base-patch16-224

# 検索設定
MAX_RESULTS=20
RANDOM_IMAGE_COUNT=6
```

### 3. ディレクトリ構造の準備

```bash
# 必要なディレクトリを作成
mkdir -p api/{photo,query_image,query_wait,add,features}

# サンプルデータの配置
# api/hakodate_result.json にデータベースファイルを配置
# api/photo/ に観光地画像を配置
# api/query_image/ にサンプル画像を配置
```

### 4. 特徴量の事前計算

```bash
# テキスト特徴量の抽出
python -c "
from text.bert_utils import FeatureManager
manager = FeatureManager()
manager.load_or_extract_features('api/hakodate_result.json', 'api/features', force_extract=True)
"

# 画像特徴量の抽出
python -c "
from image.ViT.vit_utils import extract_and_save_features
extract_and_save_features('api/hakodate_result.json', 'api/features')
"
```

### 5. サーバー起動

```bash
# 開発サーバーの起動
python main.py

# または uvicorn を直接使用
uvicorn main:app --host 0.0.0.0 --port 3110 --reload
```

## Docker での実行

```bash
# イメージのビルド
docker build -t kankodori-backend .

# コンテナの実行
docker run -p 3110:3110 kankodori-backend

# または docker-compose を使用
docker-compose up
```

## 設定

### 主要設定項目

- **パス設定**: データファイル、画像ディレクトリのパス
- **モデル設定**: BERT、ViT モデルの設定
- **検索設定**: 結果件数、重み付けパラメータ
- **サーバー設定**: ホスト、ポート、CORS 設定

### カスタマイズ

設定は `config/settings.py` で一元管理されており、環境変数での上書きが可能です。

## 開発情報

### コード品質

- **型ヒント**: 全関数に型アノテーション
- **ドキュメント**: 包括的な docstring
- **エラーハンドリング**: 適切な例外処理
- **ログ**: 詳細なログ出力

### テスト

```bash
# テストの実行
python -m pytest test/

# カバレッジレポート
python -m pytest --cov=. test/
```

### パフォーマンス

- **並列処理**: BERT 特徴量抽出の並列化
- **キャッシュ**: データとモデルのキャッシュ機能
- **最適化**: メモリ効率的な処理

## トラブルシューティング

### よくある問題

1. **メモリ不足**: 大量の画像処理時

   - バッチサイズの調整
   - 並列処理数の削減

2. **モデル読み込みエラー**:

   - インターネット接続の確認
   - モデルキャッシュのクリア

3. **ファイルパスエラー**:
   - 設定ファイルのパス確認
   - ディレクトリ権限の確認

### ログ確認

```bash
# アプリケーションログ
tail -f logs/app.log

# エラーログ
tail -f logs/error.log
```

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 貢献

プルリクエストや Issue の報告を歓迎します。開発に参加する場合は、コーディング規約に従ってください。

## 更新履歴

### v2.0.0 (リファクタリング版)

- モジュール構造の大幅改善
- 設定管理の一元化
- 型安全性の向上
- エラーハンドリングの強化
- ドキュメントの充実

### v1.0.0 (初期版)

- 基本的な検索機能
- FastAPI 実装
- BERT/ViT 統合
