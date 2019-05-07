import platform
import argparse



if(platform.system() == 'Linux'):
    COCO_IMG_PATH = "Data_Share/Datas/VQA_COCO"
    YOLO_LABEL_PATH = "Data_Share/Datas/VQA_COCO"
    addr_prefix="../../"+COCO_IMG_PATH
    addr_label_prefix="../../"+YOLO_LABEL_PATH
elif (platform.system() == 'Windows'):
    COCO_IMG_PATH = "D:\\Data_Share\Datas\VQA_COCO"
    YOLO_LABEL_PATH = "D:\\Data_Share\Datas\VQA_COCO"
    addr_prefix=COCO_IMG_PATH
    addr_label_prefix=YOLO_LABEL_PATH

