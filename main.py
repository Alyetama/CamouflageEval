#!/usr/bin/env python
# coding: utf-8

import copy
import json
import sys
import warnings
from pathlib import Path, PurePath

import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np
from colorthief import ColorThief
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from skimage import color
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def run(predictor, img_path, crop_pixels=70):
    output_image_path = "out.png"
    img_path = str(img_path)
    img = cv2.imread(img_path)

    # crop n pixels from the bottom of the image.
    img = img[:-crop_pixels, :, :]

    outputs = predictor(img)

    # get the predicted mask
    v = Visualizer(img[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2)
    v.draw_instance_predictions(outputs["instances"].to("cpu"))
    mask = outputs["instances"].pred_masks[0].cpu().numpy()
    _mask = copy.deepcopy(mask)

    # crop the image using the bounding box first.
    x1, y1, x2, y2 = outputs["instances"].pred_boxes[0].tensor.cpu().numpy()[0]
    img = img[int(y1):int(y2), int(x1):int(x2)]
    mask = mask[int(y1):int(y2), int(x1):int(x2)]
    mask = np.stack((mask, mask, mask), axis=2)
    img = img * mask

    # then crop the image using the predicted mask and save as png.
    img = img * mask
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # noqa
    img[:, :, 3] = mask[:, :, 0] * 255
    cv2.imwrite(output_image_path, img)  # noqa

    # calculate the dominant color of the cropped image use colorthief
    color_thief = ColorThief(output_image_path)
    dominant_color_1 = color_thief.get_color(quality=1)
    _dominant_color_1 = [x / 255 for x in dominant_color_1]
    dominant_color_1_lab = color.rgb2lab([[_dominant_color_1]])
    print(dominant_color_1)

    # # replace the detected object with empty pixels then save the image.
    img = cv2.imread(img_path)  # noqa
    # replace the contour of the mask with transparent pixels (alpha=0),
    # then save the image as "bg.png" keeping the alpha channel (4th channel).
    mask = np.stack((_mask, ) * 3, axis=-1)
    mask = mask.astype(np.uint8)
    mask = mask * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = cv2.bitwise_not(mask)
    img = img[:-crop_pixels, :, :]
    img[mask == 0] = 0

    # use 1.5 the size of the bbox around the detected, object then crop,
    # the image and save it as "bg.png" while keeping the alpha channel.
    img = img[int(y1 - (y2 - y1) * 0.5):int(y2 + (y2 - y1) * 0.5),
              int(x1 - (x2 - x1) * 0.5):int(x2 + (x2 - x1) * 0.5)]
    cv2.imwrite('bg.png', img)  # noqa

    color_thief = ColorThief('bg.png')
    dominant_color_2 = color_thief.get_color(quality=1)
    _dominant_color_2 = [x / 255 for x in dominant_color_2]
    dominant_color_2_lab = color.rgb2lab([[_dominant_color_2]])
    print(dominant_color_2)

    # calculate the Delta-E between dominant_color_1 and dominant_color_2.
    delta_E = colour.delta_E(dominant_color_1_lab,
                             dominant_color_2_lab,
                             method='CIE 2000')
    delta_E = round(delta_E, 2)
    print(delta_E)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(cv2.imread(output_image_path)[:, :, ::-1])
    ax[0, 1].imshow(cv2.imread('bg.png')[:, :, ::-1])
    ax[1, 0].add_patch(plt.Rectangle((0, 0), 1, 1, fc=_dominant_color_1))
    ax[1, 1].add_patch(plt.Rectangle((0, 0), 1, 1, fc=_dominant_color_2))
    fig.suptitle(f"Delta-E: {delta_E}")
    # remove all ticks and labels.
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    ax[0, 0].set_xlabel('Detected animal', fontsize=10)
    ax[0, 1].set_xlabel('Background', fontsize=10)
    ax[1, 0].set_xlabel(
        f"RGB: ({dominant_color_1[0]:.2f}, {dominant_color_1[1]:.2f}, "
        f"{dominant_color_1[2]:.2f})",
        fontsize=10)
    ax[1, 1].set_xlabel(
        f"RGB: ({dominant_color_2[0]:.2f}, {dominant_color_2[1]:.2f}, "
        f"{dominant_color_2[2]:.2f})",
        fontsize=10)

    plt.savefig(f'tmp/{Path(img_path).name}')
    plt.close()
    return delta_E


if __name__ == '__main__':
    Path('tmp').mkdir(exist_ok=True)
    input_folder = sys.argv[1]
    imgs = list(Path(input_folder).rglob("*.JPG"))

    predictor = load_model()
    scores = {}

    for img_path in tqdm(imgs):
        try:
            delta_E = run(predictor, img_path, 200)
            im_path = str(PurePath(*img_path.parts[1:]))
            scores[im_path] = delta_E
        except Exception as e:
            print(e)

    with open('scores.json', 'w') as f:
        json.dump(scores, f)
