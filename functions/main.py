"""
観光地推薦システムのメインAPIサーバー

このモジュールはFastAPIを使用して観光地推薦システムのバックエンドAPIを提供します。
テキストと画像の両方を使用した類似性検索、画像アップロード、データ追加機能を提供します。
"""

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import socket
from typing import Dict, Any

from config.settings import get_settings
from utils.file_utils import FileManager, DataManager
from result import calc
from add_data import add_data

# 設定の取得
settings = get_settings()

app = FastAPI(
    title="観光地推薦システムAPI",
    description="テキストと画像を用いた観光地推薦システム",
    version="1.0.0"
)

# ================================================================================
# ネットワーク設定とCORS設定
# ================================================================================

def get_ip_address() -> str:
    """
    現在のホストのIPアドレスを取得する

    Returns:
        str: ホストのIPアドレス
    """
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception as e:
        print(f"IPアドレス取得エラー: {e}")
        return "localhost"

# IPアドレスの取得と表示
ip_address = get_ip_address()
print(f"Current IP Address: {ip_address}")

# 許可するオリジンの設定
origins = [
    "https://kankodori.web.app",
    f"http://{ip_address}:5173"
]

# CORS設定（設定ファイルから取得）
cors_config = settings.get_cors_config()
app.add_middleware(
    CORSMiddleware,
    **cors_config
)

@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    """
    カスタムCORSヘッダーミドルウェア

    静的ファイルアクセス時にもCORSヘッダーを追加し、
    ngrokブラウザ警告をスキップするためのヘッダーを設定
    """
    response = await call_next(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept, Authorization, ngrok-skip-browser-warning'
    response.headers['ngrok-skip-browser-warning'] = 'true'
    response.headers['User-Agent'] = 'Custom-User-Agent'
    return response

# ================================================================================
# 静的ファイル配信設定
# ================================================================================

def setup_static_files():
    """
    静的ファイル配信を設定する
    """
    # 静的ファイルの配信設定
    app.mount("/api/photo", StaticFiles(directory=settings.paths.PHOTO_DIR), name="photo_files")
    app.mount("/api/query_image", StaticFiles(directory=settings.paths.QUERY_IMAGE_DIR), name="query_image_files")

# 静的ファイル設定の実行
setup_static_files()

# ================================================================================
# データマネージャーの初期化
# ================================================================================

# データマネージャーのインスタンス化
data_manager = DataManager(settings.paths.HAKODATE_JSON)

# ================================================================================
# APIエンドポイント
# ================================================================================

@app.get("/")
def health_check() -> Dict[str, str]:
    """
    ヘルスチェック用エンドポイント

    Returns:
        Dict[str, str]: サーバーの稼働状況
    """
    return {"status": "healthy", "message": "観光地推薦システムAPIが正常に動作しています"}

@app.post("/upload")
async def upload_and_search(
    request: Request,
    file: UploadFile = File(...),
    text: str = Form(...),
    range: int = Form(...)
) -> Dict[str, Any]:
    """
    画像とテキストによる観光地検索

    Args:
        request: HTTPリクエストオブジェクト
        file: アップロードされた画像ファイル
        text: 検索テキスト
        range: テキスト（0）と画像（100）の重み付け（0-100）

    Returns:
        Dict[str, Any]: 検索結果と画像URL付きの観光地リスト

    Raises:
        HTTPException: 処理エラーの場合
    """
    try:
        # アップロードされた画像ファイルの保存
        if file.filename != "null":
            file_path = os.path.join(settings.paths.QUERY_WAIT_DIR, file.filename)

            if not FileManager.save_uploaded_file(file, file_path):
                raise HTTPException(status_code=500, detail="ファイル保存に失敗しました")

        # 受信データのログ出力
        print(f"検索テキスト: {text}")
        print(f"重み付け範囲: {range}")
        print(f"アップロードファイル: {file.filename}")

        # 類似度計算の実行
        labels, features, sorted_results = calc(range, file.filename, text)

        # 結果に画像URLを追加
        results_with_urls = []
        max_results = settings.search.MAX_RESULTS

        for item in sorted_results[:max_results]:
            image_url = f"{request.url.scheme}://{request.url.netloc}/api/photo/{item['id']}.jpg"
            results_with_urls.append({**item, "image_url": image_url})

        return {
            "range": range,
            "result": results_with_urls,
            "total_found": len(sorted_results)
        }

    except Exception as e:
        error_message = f"検索処理に失敗しました: {str(e)}"
        print(f"検索処理中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/query_image")
async def get_random_sample_images() -> Dict[str, Any]:
    """
    ランダムなサンプル画像を取得

    クエリ用画像ディレクトリからランダムに画像を選択して返す

    Returns:
        Dict[str, Any]: ランダムに選択された画像ファイル名のリスト

    Raises:
        HTTPException: 処理エラーの場合
    """
    try:
        random_images = FileManager.get_random_images(
            settings.paths.QUERY_IMAGE_DIR,
            settings.search.RANDOM_IMAGE_COUNT
        )

        print(f"ランダム画像を取得しました: {len(random_images)}枚")
        return {"images": random_images}

    except Exception as e:
        error_message = f"画像取得に失敗しました: {str(e)}"
        print(f"ランダム画像取得中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/query_image/{image_name}")
async def get_sample_image(image_name: str):
    """
    指定されたサンプル画像ファイルを取得

    Args:
        image_name: 取得する画像ファイル名

    Returns:
        FileResponse: 画像ファイル、または404エラー

    Raises:
        HTTPException: ファイルが見つからない場合
    """
    image_path = os.path.join(settings.paths.QUERY_IMAGE_DIR, image_name)

    if os.path.exists(image_path):
        return FileResponse(image_path, media_type='image/jpeg')
    else:
        raise HTTPException(status_code=404, detail="指定された画像が見つかりません")

@app.post("/add_place")
async def add_new_place(
    file: UploadFile = File(...),
    name: str = Form(...),
    location: str = Form(...),
    description: str = Form(...),
    tags: str = Form(...)
) -> Dict[str, str]:
    """
    新しい観光地をデータベースに追加

    Args:
        file: 観光地の画像ファイル
        name: 観光地名
        location: 場所
        description: 説明
        tags: タグ（カンマ区切り）

    Returns:
        Dict[str, str]: 追加結果メッセージ

    Raises:
        HTTPException: 処理エラーの場合
    """
    try:
        # 画像ファイルの保存
        file_path = os.path.join(settings.paths.ADD_DIR, file.filename)

        if not FileManager.save_uploaded_file(file, file_path):
            raise HTTPException(status_code=500, detail="ファイル保存に失敗しました")

        # 受信データのログ出力
        print(f"新規観光地追加:")
        print(f"  ファイル名: {file.filename}")
        print(f"  観光地名: {name}")
        print(f"  場所: {location}")
        print(f"  説明: {description}")
        print(f"  タグ: {tags}")

        # データ追加処理の実行
        success = add_data(file.filename, name, location, description, tags)

        if success:
            return {"message": "観光地が正常に追加されました"}
        else:
            raise HTTPException(status_code=500, detail="観光地の追加に失敗しました")

    except HTTPException:
        raise
    except Exception as e:
        error_message = f"観光地の追加に失敗しました: {str(e)}"
        print(f"観光地追加中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/stats")
async def get_system_stats() -> Dict[str, Any]:
    """
    システム統計情報を取得

    Returns:
        Dict[str, Any]: システム統計情報
    """
    try:
        photos = data_manager.get_photos()

        # 一時ファイルのクリーンアップ
        cleaned_count = FileManager.clean_temp_files(settings.paths.QUERY_WAIT_DIR)

        return {
            "total_photos": len(photos),
            "temp_files_cleaned": cleaned_count,
            "system_status": "healthy"
        }

    except Exception as e:
        print(f"統計情報取得エラー: {e}")
        raise HTTPException(status_code=500, detail="統計情報の取得に失敗しました")

# ================================================================================
# サーバー起動設定
# ================================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"サーバーを起動します: http://{ip_address}:{settings.server.PORT}")
    uvicorn.run(
        app,
        host=ip_address,
        port=settings.server.PORT,
        reload=settings.server.RELOAD
    )
