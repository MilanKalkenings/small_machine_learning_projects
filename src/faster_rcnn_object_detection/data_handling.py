import os
import xml.etree.ElementTree as ET
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


project_dir = "drive/MyDrive/uwsagp/"

def load_data(annot_path: str, img_squeeze: int, clean_dir: str):
    trans = Compose([Resize(size=int(2736 / img_squeeze)), ToTensor()])
    imgs = []
    img_paths = []
    boxes = []

    # litter
    xml_file_path = annot_path
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for image in root:
        img_path = project_dir + "data/litter/" + [image.attrib["name"]][0]
        img_paths.append(img_path)
        imgs.append(torch.unsqueeze(trans(Image.open(img_path)), dim=0))
        img_boxes = []
        for box in image:
            attr = box.attrib
            img_boxes.append([float(attr["xtl"])/img_squeeze,
                              float(attr["ytl"])/img_squeeze,
                              float(attr["xbr"])/img_squeeze,
                              float(attr["ybr"])/img_squeeze])
        boxes.append(torch.tensor(img_boxes))

    # clean
    img_files_clean = os.listdir(clean_dir)
    for file in img_files_clean:
        img_path = project_dir + "data/clean/" + file
        img_paths.append(img_path)
        imgs.append(torch.unsqueeze(trans(Image.open(img_path)), dim=0))
        boxes.append(torch.empty((0, 4)))
    return imgs, boxes, img_paths
    