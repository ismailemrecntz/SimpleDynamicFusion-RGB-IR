import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def calculate_mean_std(image_dir):
    """
    Calculate the mean and standard deviation of a dataset.
    Args:
        image_dir (str): Path to the directory containing images.
    Returns:
        rgb_mean (list): Mean values for RGB channels.
        rgb_std (list): Standard deviation values for RGB channels.
        ir_mean (float): Mean value for IR channel.
        ir_std (float): Standard deviation value for IR channel.
    """
    rgb_means = []
    rgb_stds = []
    ir_means = []
    ir_stds = []

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for img_path in tqdm(image_files, desc="Processing images"):
        img = np.asarray(Image.open(img_path))

        # Split RGB and IR channels
        rgb = img[:, :, :3]  # First 3 channels
        ir = img[:, :, 3] if img.shape[2] > 3 else np.zeros_like(img[:, :, 0])  # Last channel or zeros if not present

        # Normalize to [0, 1]
        rgb = rgb / 255.0
        ir = ir / 255.0

        # Calculate mean and std for RGB
        rgb_means.append(np.mean(rgb, axis=(0, 1)))
        rgb_stds.append(np.std(rgb, axis=(0, 1)))

        # Calculate mean and std for IR
        ir_means.append(np.mean(ir))
        ir_stds.append(np.std(ir))

    rgb_mean = np.mean(rgb_means, axis=0)
    rgb_std = np.mean(rgb_stds, axis=0)
    ir_mean = np.mean(ir_means)
    ir_std = np.mean(ir_stds)

    return rgb_mean, rgb_std, ir_mean, ir_std

import numpy as np
from PIL import Image 
 
# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

from PIL import Image
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize(image_name, predictions, weight_name, rgb=None, ir=None, labels=None, save_dir='runs'):
    os.makedirs(save_dir, exist_ok=True)
    palette = get_palette() 
    
    for i in range(predictions.shape[0]):
        pred = predictions[i].cpu().numpy().astype(np.uint8)
        H, W = pred.shape
        
        # --- Prediction to color ---
        pred_img = np.zeros((H, W, 3), dtype=np.uint8)
        for cid in range(len(palette)):
            pred_img[pred == cid] = palette[cid]
        pred_img = Image.fromarray(pred_img)
        
        # --- RGB ---
        if rgb is not None:
            rgb_np = rgb[i].permute(1, 2, 0).cpu().numpy()
            rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
            rgb_img = Image.fromarray((rgb_np * 255).astype(np.uint8))
        else:
            rgb_img = Image.new("RGB", (W, H), (0, 0, 0))
        
        # --- IR ---
        if ir is not None:
            ir_np = ir[i].squeeze().cpu().numpy()
            ir_np = (ir_np - ir_np.min()) / (ir_np.max() - ir_np.min() + 1e-8)
            ir_gray = Image.fromarray((ir_np * 255).astype(np.uint8)).convert("L")
            ir_img = ir_gray.convert("RGB")
        else:
            ir_img = Image.new("RGB", (W, H), (0, 0, 0))
        
        # --- Ground Truth ---
        if labels is not None:
            gt_np = labels[i].cpu().numpy().astype(np.uint8)
            gt_img = np.zeros((H, W, 3), dtype=np.uint8)
            for cid in range(len(palette)):
                gt_img[gt_np == cid] = palette[cid]
            gt_img = Image.fromarray(gt_img)
        else:
            gt_img = Image.new("RGB", (W, H), (0, 0, 0))
        
        # --- Combine (RGB | IR | Pred | GT) ---
        combined = Image.new('RGB', (W * 4, H))
        combined.paste(rgb_img, (0, 0))
        combined.paste(ir_img, (W, 0))
        combined.paste(pred_img, (2 * W, 0))
        combined.paste(gt_img, (3 * W, 0))
        
        save_path = os.path.join(save_dir, f'Vis_{weight_name}_{image_name[i]}.png')
        combined.save(save_path)
        print(f"[SAVED] {save_path}")

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class):
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/(TP+FP)
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/(TP+FN)
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/(TP+FP+FN)

        f1score=2*precision_per_class*recall_per_class/(precision_per_class+recall_per_class)

    return precision_per_class, recall_per_class, iou_per_class, f1score

if __name__ == '__main__':

    # Example usage
    image_dir = '/datavolume/data/emrecanitez/Datasets/MFNet/images/' 
    rgb_mean, rgb_std, ir_mean, ir_std = calculate_mean_std(image_dir)
    print("RGB Mean:", rgb_mean)
    print("RGB Std:", rgb_std)
    print("IR Mean:", ir_mean)
    print("IR Std:", ir_std)

