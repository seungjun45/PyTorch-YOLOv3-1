import platform
import argparse

COCO_IMG_PATH="D:\\Data_Share\Datas\VQA_COCO"
YOLO_LABEL_PATH="D:\\Data_Share\Datas\VQA_COCO"

if(platform.system() == 'Linux'):
    addr_prefix="../../"+COCO_IMG_PATH[4:]
    addr_label_prefix="../../"+YOLO_LABEL_PATH[4:]
elif (platform.system() == 'Windows'):
    addr_prefix=COCO_IMG_PATH
    addr_label_prefix=YOLO_LABEL_PATH

