from PIL import Image
import os
import json
import random
import MeCab
import numpy as np
from image.ViT.vit_utils import extract_features
from text.bert_utils import extract_features, process_photo

mode=1

def add_data(image,name,location,review,tag):
    if mode==1:
        return
