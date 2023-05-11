import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama', refine=True)
all_scenes = ['DayaTemple', 'HaiyanHall', 'Library', 'MemorialHall', 'Museum', 'PeonyGarden', 'ScienceSquare', 'theOldGate']
if len(sys.argv) < 2:
    raise ValueError("Usage: python3 ./inpainting.py <scene_id>")
scene_id = int(sys.argv[1])
scale_rate = 4
scene_name = all_scenes[scene_id]

renders_path = "../renders"
result_path = "."

def load_image(imfile, factor=1):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if factor != 1:
        img = cv2.resize(img, (img.shape[1] // factor, img.shape[0] // factor))
    return img

for img_name in tqdm(os.listdir(os.path.join(renders_path, scene_name))):
    img_id = img_name[:8]
    img_path = os.path.join(renders_path, scene_name, f"{img_id}.jpg")
    src_img = load_image(img_path, scale_rate)

    final_img_np = np.array(Image.open(os.path.join(result_path, "reproject", scene_name, f"{img_id}.png"))).astype(np.uint8)
    final_cnt_np = np.array(Image.open(os.path.join(result_path, "reproject", scene_name, f"{img_id}_cnt.png"))).astype(np.uint8)
    img_diff = np.array(Image.open(os.path.join(result_path, "reproject", scene_name, f"{img_id}_diff.png"))).astype(np.uint8)

    ori_mask = img_diff > 200
    final_img_np[ori_mask, :] = src_img[ori_mask, :]

    input = {
        'img': Image.fromarray(final_img_np.astype('uint8')),
        'mask': Image.fromarray((final_cnt_np < 1).astype('uint8')),
    }
    result = inpainting(input)
    final_img_np = result[OutputKeys.OUTPUT_IMG][...,::-1]

    src_img = load_image(img_path)
    final_img_np = cv2.resize(final_img_np, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    img_diff = cv2.resize(img_diff.astype('uint8'), (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    ori_mask = img_diff > 179
    final_img_np[ori_mask, :] = src_img[ori_mask, :]

    folder_path = os.path.join(result_path, "results", scene_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    res_path = os.path.join(folder_path, f"{img_id}.jpg")
    cv2.imwrite(res_path, final_img_np[...,::-1])
