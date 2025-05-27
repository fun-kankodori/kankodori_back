#dockerイメージを指定。
FROM python:3.10.9
RUN apt-get -y update && apt-get -y upgrade
#コンテナ内での作業ディレクトリを指定。
WORKDIR /root

#requirements.txtをコンテナ内のroot配下にコピー。
COPY requirements.txt root/requirements.txt
# コンテナ起動時にモジュールをインストール。
#RUN pip install --no-cache-dir --upgrade -r root/requirements.txt
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
RUN pip install uvicorn fastapi numpy scikit-learn pillow python-multipart transformers python-dotenv googletrans==4.0.0-rc1 matplotlib argostranslate mecab-python3
#main.pyを/root/appに以下にコピー。
COPY myvenv root/myvenv

#コンテナ起動時に実行するコマンドを指定。
EXPOSE 8000

CMD ["uvicorn", "root.myvenv.main:app","--host", "0.0.0.0", "--port", "8000", "--reload"]
