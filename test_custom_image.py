import argparse
import os
import time
from os.path import isfile

import torch
import torch.optim
import numpy as np
import yaml
from PIL import Image
import cv2

from logger import Logger

from model.network import Net
from skimage.measure import label, regionprops
from utils import reverse_mapping, get_boundary_point

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
# arguments from command line
parser.add_argument('--config', default='./config.yml', help='path to config file')
parser.add_argument('--model', required=True, help='path to the pretrained model')
parser.add_argument('--test_dir', required=True, help="path to store test files")
parser.add_argument('--align', default=False, action='store_true', help='whether to use align')
parser.add_argument('--tmp', default='./tmp', help='temp directory')
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.load(open(args.config), Loader=yaml.FullLoader)

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))


def main():
    logger.info(args)

    model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"],
                backbone=CONFIGS["MODEL"]["BACKBONE"])
    model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    if args.model:
        if isfile(args.model):
            logger.info("=> loading pretrained model '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            if 'state_dict' in checkpoint.keys():
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("=> loaded checkpoint '{}'".format(args.model))
        else:
            logger.info("=> no pretrained model found at '{}'".format(args.model))

    logger.info("Start testing.")
    # switch to evaluate mode
    model.eval()
    test_files = os.listdir(args.test_dir)
    with torch.no_grad():
        for _, fn in enumerate(test_files):
            t = time.time()
            image = Image.open(os.path.join(args.test_dir, fn))
            image = image.resize(size=(400, 400))
            image = np.array(image)
            size = image.shape[:2]

            image_tensor = torch.from_numpy(image).float() / 255.
            image_tensor = torch.transpose(image_tensor, 0, 2)
            image_tensor = torch.transpose(image_tensor, 1, 2)
            image_tensor = image_tensor[None, :, :, :]
            image_tensor = image_tensor.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            key_points = model(image_tensor)

            key_points = torch.sigmoid(key_points)
            t = time.time() - t
            print("time:", t)

            binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)

            b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"],
                                       size=size)
            print("b_points:", b_points)
            for i in range(len(b_points)):
                y1 = int(np.round(b_points[i][0]))
                x1 = int(np.round(b_points[i][1]))
                y2 = int(np.round(b_points[i][2]))
                x2 = int(np.round(b_points[i][3]))
                if x1 == x2:
                    angle = -np.pi / 2
                else:
                    angle = np.arctan((y1 - y2) / (x1 - x2))
                (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                b_points[i] = (y1, x1, y2, x2)

            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for (y1, x1, y2, x2) in b_points:
                img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=int(0.01 * max(size[0], size[1])))
            cv2.imshow("Main", img)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
