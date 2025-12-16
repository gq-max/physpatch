import os
import argparse
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass

from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto


def initialize_models(sam_ckpt_path):
    print("Initializing SAM model...")
    model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).eval().cuda()
    print("Model ready.")
    return None, model_sam, None  # placeholders for semsam/seem


def process_image(image_path, model_sam, args):
    image = Image.open(image_path).convert('RGB')

    if args.granularity >= 2.5:
        model_name = 'sam'
    else:
        raise NotImplementedError("Only SAM is supported in this script version.")

    label_mode = 'a' if args.label_mode == 'Alphabet' else '1'
    text_size = 900
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        output, anns = inference_sam_m2m_auto(
            model_sam, image, text_size, label_mode, args.alpha, args.anno_mode
        )

    labels = []
    H, W = anns[0]["segmentation"].shape
    for ann in anns:
        cy, cx = center_of_mass(ann["segmentation"])
        x_norm = (2 * cx / W) - 1
        y_norm = (2 * cy / H) - 1
        labels.append((x_norm, y_norm))

    return output, labels


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.label_dir, exist_ok=True)

    _, model_sam, _ = initialize_models(args.sam_ckpt)

    for filename in sorted(os.listdir(args.input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, filename)
            try:
                result, labels = process_image(image_path, model_sam, args)

                output_path = os.path.join(args.output_dir, filename)
                Image.fromarray(result).save(output_path)

                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(args.label_dir, label_filename)
                with open(label_path, 'w') as f:
                    for x_norm, y_norm in labels:
                        f.write(f"{x_norm:.4f}, {y_norm:.4f}\n")

                print(f"Processed {filename} successfully.")

            except Exception as e:
                print(f"[Error] {filename}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Process Images with SAM and Save Normalized Labels')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save segmented mask outputs')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory to save normalized center labels')
    parser.add_argument('--sam_ckpt', type=str, required=True,
                        help='Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)')
    parser.add_argument('--granularity', type=float, default=2.6,
                        help='Granularity level: [1,1.5)=SEEM, [1.5,2.5)=SemanticSAM, [2.5,3]=SAM')
    parser.add_argument('--mode', choices=['Automatic', 'Interactive'], default='Automatic',
                        help='Segmentation mode (batch only supports Automatic)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Mask transparency [0~1]')
    parser.add_argument('--label_mode', choices=['Number', 'Alphabet'], default='Number',
                        help='Label display mode')
    parser.add_argument('--anno_mode', nargs='+', default=['Mask', 'Mark'],
                        choices=['Mask', 'Box', 'Mark'],
                        help='Annotation types to include')

    args = parser.parse_args()
    args.mode = 'Automatic'

    main(args)
