import json
import pickle
import os
import pdb

data_root='/mnt/server6_hard2/junyoung/PJT/GTA5/'
json_path=data_root+'instances_GTA_train_small_2.json'
img2anno_file='img2anno.pkl'

json_=json.load(open(json_path))
image_length=len(json_['images'])
anno_length=len(json_['annotations'])
imgid_2_annoid={}
for anno in json_['annotations']:
    if(not(anno['image_id'] in imgid_2_annoid)):
        imgid_2_annoid[anno['image_id']]=[]
    imgid_2_annoid[anno['image_id']].append(anno['id'])

pickle.dump(imgid_2_annoid, open(img2anno_file, 'wb'))
