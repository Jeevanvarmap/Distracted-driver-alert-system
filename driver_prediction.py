import os
import json
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil

from keras.preprocessing import image                  
from tqdm.notebook import tqdm
from PIL import ImageFile                            

BASE_MODEL_PATH = os.path.join(os.getcwd())
PICKLE_DIR = os.path.join(os.getcwd())
JSON_DIR = os.path.join(os.getcwd(),"model_data/json_files")

if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)

BEST_MODEL = os.path.join(BASE_MODEL_PATH,"model_data/model.hdf5")
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR,"model_data/labels_list.pkl"),"rb") as handle:
    labels_id = pickle.load(handle)

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True 

def predict_result(image_tensor):

    ypred_test = model.predict(image_tensor,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)
    # print(ypred_class)

    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    ypred_class = int(ypred_class)
    #print(id_labels[ypred_class])

    class_name = dict()
    class_name["c0"] = "SAFE_DRIVING"
    class_name["c1"] = "TEXTING_RIGHT"
    class_name["c2"] = "TALKING_PHONE_RIGHT"
    class_name["c3"] = "TEXTING_LEFT"
    class_name["c4"] = "TALKING_PHONE_LEFT"
    class_name["c5"] = "OPERATING_RADIO"
    class_name["c6"] = "DRINKING"
    class_name["c7"] = "REACHING_BEHIND"
    class_name["c8"] = "HAIR_AND_MAKEUP"
    class_name["c9"] = "TALKING_TO_PASSENGER"


    with open(os.path.join(JSON_DIR,'class_name_map.json'),'w') as secret_input:
        json.dump(class_name,secret_input,indent=4,sort_keys=True)

    with open(os.path.join(JSON_DIR,'class_name_map.json')) as secret_input:
        info = json.load(secret_input)
        label = info[id_labels[ypred_class]]
        #print(label)
    
    return label