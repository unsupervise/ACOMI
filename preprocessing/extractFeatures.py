import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from vit_keras import vit, utils
import cv2
import tensorflow as tf
import tensorflow_hub as hub

df = pd.read_csv("dataset.csv")
df.drop("Unnamed: 0", inplace = True, axis = 1)

def encode_with_vit():
    
    image_size = 224
    model_handle = vit.vit_b16
    
    model = model_handle(image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=False,
    pretrained_top=False)
    
    for i in tqdm(range(len(df))):
        views = []
        paths = df.iloc[i,:].tolist()
        for path in paths:
            views.append(cv2.imread(+path))
        
        views = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), views)
        views = map(lambda x: cv2.resize(x, (image_size, image_size),), views)
        views = map(lambda x: vit.preprocess_inputs(x).reshape(1, image_size, image_size, 3), views)
        views = map(lambda x: model.predict(x), views)
        views = map(lambda x: x.tolist(), views)
        with open(saveFolder + "/{}".format(i), "w") as fp:
            json.dump(list(views), fp)
            

saveFolder = "extractedFeatures/"
if(not os.path.exists(saveFolder)):
    os.makedirs(saveFolder)
print("Extracting features using vit_b16...")
encode_with_vit()


   
