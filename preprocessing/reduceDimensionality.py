import json
import os
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("d", help="n_components", type = int)
args = parser.parse_args()
d = args.d

folder = r"extractedFeatures/"
df = []

for filename in tqdm(natsorted(os.listdir(folder), key= lambda x :x.lower())):
    file = open(os.path.join(folder, filename))
    data = json.load(file)
    df = df + [np.array(x) for x in data]
    
df= np.array(df).reshape(108000,768)
pca = PCA(n_components=d)
reducedData = pca.fit_transform(df)
reducedDataList = [list(vec) for vec in reducedData]
item = []
saveFolder = "dimensionalityReduction/reducedData/pca{}".format(d)

if(not os.path.exists(saveFolder)):
    os.makedirs(saveFolder)

for i, vec in tqdm(enumerate(reducedDataList)):
    item.append(vec)
    if (((i+1) % 108) == 0 ):
        with open(saveFolder + "/{}.json".format(int((i+1) / 108)), "w") as wp:
            json.dump(item, wp) 
        item = []
        
print("PCA performed succefully!")
