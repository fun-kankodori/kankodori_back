from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil

from pydantic_core import Url
from result import calc
import random
from add_data import add_data
import socket

app = FastAPI()

def get_ip_address():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

ip_address = get_ip_address()
print(f"Current IP Address: {ip_address}")

origins = [
    "https://kankodori.web.app"
    f"http://{ip_address}:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
)

# カスタムミドルウェアを追加して静的ファイルにCORSヘッダーを追加
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept, Authorization, ngrok-skip-browser-warning'
    response.headers['ngrok-skip-browser-warning'] = 'true'
    response.headers['User-Agent'] = 'Custom-User-Agent'
    #response.headers['User-Agent'] = 'ngrok-skip-browser-warning'
    return response


api_photo=os.path.join(os.path.dirname(__file__), "api/photo")
os.makedirs(api_photo, exist_ok=True)
api_query_image=os.path.join(os.path.dirname(__file__), "api/query_image")
os.makedirs(api_query_image, exist_ok=True)
api_query_wait=os.path.join(os.path.dirname(__file__), "api/query_wait")
os.makedirs(api_query_wait, exist_ok=True)

# 静的ファイルの提供を設定
app.mount("/api/photo", StaticFiles(directory=api_photo), name="refiles")
app.mount("/api/query_image", StaticFiles(directory=api_query_image), name="query_image")

@app.get("/")
def Hello():
    return {"Hello": "World!"}

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), text: str = Form(...), range: int = Form(...)):
    try:
        # 画像ファイルの保存
        if file.filename != "null":
            file_location = os.path.join(api_query_wait, file.filename)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        # 受け取ったデータの処理
        print(f"Received text: {text}")
        print(f"Received range: {range}")
        print(f"Received file: {file.filename}")
        labels, features, sort = calc(range, file.filename, text)
        # 画像処理の呼び出し
        # process_images(file_location)

        # 画像のURLを返す
        results_with_urls = []
        for item in sort[:20]:
            image_url = f"{request.url.scheme}://{request.url.netloc}/api/photo/{item['id']}.jpg"
            results_with_urls.append({**item, "image_url": image_url})

        return {"range": range, "result": results_with_urls}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": str(e)}

# 画像が保存されているディレクトリのパス
IMAGE_DIR = "api/photo"

@app.get("/api/query_image")
async def get_random_images():
    print("画像を取得します")
    try:
        image_files = os.listdir(api_query_image)
        random_images = random.sample(image_files, 6)
        return {"images": random_images}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": str(e)}

@app.get("/api/query_image/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(api_query_image, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type='image/jpeg')
    else:
        return {"error": "Image not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,  host=ip_address, port=3110)

@app.post("/add_place")
async def add_place(file: UploadFile = File(...), name: str = Form(...), location: str = Form(...), description: str = Form(...), tags: str = Form(...)):
    try:
        # 画像ファイルの保存
        api_add=os.path.join(os.path.dirname(__file__), "api/add")
        file_location = os.path.join(api_add, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Received file: {file.filename}")
        print(f"Received name: {name}")
        print(f"Received location: {location}")
        print(f"Received description: {description}")
        print(f"Received tags: {tags}")

        add_data(file.filename, name, location, description, tags)
        return {"message": "観光地が追加されました"}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": str(e)}

