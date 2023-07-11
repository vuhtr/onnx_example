import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import cv2
import time
from utils import *
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to onnx model")
    parser.add_argument("--input_dir", required=True, help="input image path")
    parser.add_argument("--output_dir", default=None, help="output image path")
    parser.add_argument("--infer_size", default=800, type=int, help="inference size")
    parser.add_argument('--common_conf_thres', default=0.35, type=float, help='common confidence threshold')
    parser.add_argument('--person_conf_thres', default=0.25, type=float, help='person confidence threshold')
    args = parser.parse_args()
    return args

args = get_args()

class_dict = {}
with open('classes.txt', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        class_dict[i] = line.strip()

model_path = args.model_path
infer_size = args.infer_size

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

ort_sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = ort_sess.get_inputs()[0].name
output_names = [x.name for x in ort_sess.get_outputs()]


avg_inference_time = 0

for input_path in img_paths:
    img = Image.open(input_path).convert('RGB')
    # add resize to 1024 (like mobile)
    # img = resize_keep_ratio(img, 1024, mode=Image.BILINEAR)
    start = time.time()
    input_tensor, padded_img, pad_left, pad_top = preprocess(img, max_size=infer_size, pad_val=PAD_VAL)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    print('Preprocess time:', time.time() - start)

    start = time.time()
    result = ort_sess.run(output_names, {input_name: input_tensor})
    inference_time = time.time() - start
    print('Inference time:', inference_time)

    start = time.time()

    dets = result[0][0]
    scores = [x[-1] for x in dets]
    bboxes = [x.astype(np.int32)[:-1] for x in dets]
    labels = result[1][0].astype(np.int8).tolist()
    masks = [mask for mask in result[2][0]]

    # refine coors to original image
    orig_width, orig_height = img.size
    selected_idx = []
    for i in range(len(bboxes)):
        if scores[i] >= COMMON_CONF_THRES or (labels[i] == 0 and scores[i] >= PERSON_CONF_THRES):
            x1, y1, x2, y2 = bboxes[i]
            x1 = min(max(x1, pad_left), infer_size - pad_left - 1)
            y1 = min(max(y1, pad_top), infer_size - pad_top - 1)
            x2 = min(max(x2, pad_left), infer_size - pad_left - 1)
            y2 = min(max(y2, pad_top), infer_size - pad_top - 1)

            if x1 >= x2 or y1 >= y2:
                continue

            cur_mask = masks[i][y1:y2, x1:x2]
            bboxes[i][2] = x1 + cur_mask.shape[1]
            bboxes[i][3] = y1 + cur_mask.shape[0]

            # binary mask
            cur_mask = np.where(cur_mask > 0.5, 1, 0).astype(np.uint8)

            # calc original coors
            bboxes[i][0] -= pad_left
            bboxes[i][1] -= pad_top
            bboxes[i][2] -= pad_left
            bboxes[i][3] -= pad_top
            real_infer_width = infer_size - pad_left * 2
            real_infer_height = infer_size - pad_top * 2
            bboxes[i][0] = int(bboxes[i][0] * orig_width / real_infer_width)
            bboxes[i][1] = int(bboxes[i][1] * orig_height / real_infer_height)
            bboxes[i][2] = int(bboxes[i][2] * orig_width / real_infer_width)
            bboxes[i][3] = int(bboxes[i][3] * orig_height / real_infer_height)
            box_w = bboxes[i][2] - bboxes[i][0]
            box_h = bboxes[i][3] - bboxes[i][1]
            # skip too small box
            if box_w + box_h < BOX_SIZE_THRES:
                continue

            # mask of object in original image
            new_mask = cv2.resize(cur_mask, (box_w, box_h))
            masks[i] = new_mask

            selected_idx.append(i)

    print('Postprocess time:', time.time() - start)

    # visualize

    orig_img = cv2.imread(input_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    cnt = 0
    for i in selected_idx:
        mask = masks[i]
        bbox = bboxes[i]
        score = scores[i]
        label = class_dict[labels[i]]

        orig_img = cv2.rectangle(orig_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        orig_img = cv2.putText(orig_img, f'{label}', (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * (255, 0, 0)
        mask = mask.astype(np.uint8)
        mask_img = Image.new('RGB', (orig_width, orig_height), (0, 0, 0))
        mask_img.paste(Image.fromarray(mask), (bbox[0], bbox[1]))
        mask_img = np.array(mask_img)

        orig_img = cv2.addWeighted(orig_img, 1, mask_img, 0.7, 0)

        cnt += 1

    print('Detected object:', cnt)

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    output_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, output_name)

    cv2.imwrite(output_path, orig_img)

    avg_inference_time += inference_time

avg_inference_time /= len(img_paths)
print('Avg inference time:', avg_inference_time)
