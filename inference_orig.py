import cv2
import os
import numpy as np
import json

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

from utils import *
from tqdm import tqdm

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to model")
    parser.add_argument("--cfg_path", required=True, help="path to model")
    parser.add_argument("--input_dir", required=True, help="input image path")
    parser.add_argument("--output_dir", default=None, help="output image path")
    parser.add_argument('--common_conf_thres', default=0.3, type=float, help='common confidence threshold')
    parser.add_argument('--person_conf_thres', default=0.2, type=float, help='person confidence threshold')
    args = parser.parse_args()
    return args

args = get_args()

class_dict = {}
with open('classes.txt', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        class_dict[i] = line.strip()

model_path = args.model_path
cfg_path = args.cfg_path

img_paths = [os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir)]
img_paths = [x for x in img_paths]

if args.output_dir is None:
    output_dir = args.input_dir + '_out'
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

PAD_VAL = 114
BOX_SIZE_THRES = 20
MASK_COLOR = np.array([255, 0, 0])
COMMON_CONF_THRES = args.common_conf_thres
PERSON_CONF_THRES = args.person_conf_thres


#####################################################

def load_ai_detection_model(cfg_path, model_path):
    # cfg_path = "./weights/rtmdet-ins_x_8xb16-300e_coco.py"
    # weight_path = "./weights/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth"

    register_all_modules()
    model = init_detector(cfg_path, model_path, device='cpu')
    # pipeline = model.cfg.test_dataloader.dataset.pipeline
    # sz = 2048
    # pipeline[1]['scale'] = (sz, sz)
    # pipeline[2]['size'] = (sz, sz)

    return model


def read_img(img_path):
    pil_img = Image.open(img_path).convert('RGB')
    np_img = np.array(pil_img)
    return np_img


if __name__ == '__main__':
    model = load_ai_detection_model(cfg_path, model_path)

    for img_path in tqdm(img_paths):
        np_img = read_img(img_path)
        
        # Detect
        outputs = inference_detector(model, np_img).pred_instances
        instance_masks = outputs.masks.cpu().numpy()
        instance_bboxes = outputs.bboxes.cpu().numpy()
        instance_scores = outputs.scores.cpu().numpy()
        instance_labels = outputs.labels.cpu().numpy()

        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        for i in range(len(instance_masks)):
            mask = np.where(instance_masks[i] > 0.5, 1, 0).astype(np.uint8)
            bbox = instance_bboxes[i].astype(np.int32)
            score = instance_scores[i]
            label = instance_labels[i].astype(np.int32)

            if (score >= COMMON_CONF_THRES or (score >= PERSON_CONF_THRES and label == 0)):
                orig_img = cv2.rectangle(orig_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                orig_img = cv2.putText(orig_img, f'{class_dict[label]}', (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # mask = mask * 255
                mask = mask.astype(np.float32)
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * (255, 0, 0)
                mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))
                mask = mask.astype(np.uint8)
                orig_img = cv2.addWeighted(orig_img, 1, mask, 0.7, 0)


        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        output_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, orig_img)
