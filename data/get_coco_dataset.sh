#!/bin/bash

# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

# Download COCOAPI tools for Windows
wget -c https://www.dropbox.com/s/owr3uox80b5bbar/coco.zip

mkdir coco
mv coco.zip coco/
unzip -q coco/coco.zip -d coco/
rm -rf coco.zip