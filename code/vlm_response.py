import os
import sys
import gc
import argparse
from models.gpt_model import Gpt_model

def process_model(model_class, model_name, model_path, output_dir, image_list, query, stream=False):
    print(f"Loading model: {model_name}")
    if model_name == "Qwen-qvq":
        model = model_class(model_path, stream)
    else:
        model = model_class(model_path)

    os.makedirs(output_dir, exist_ok=True)
    output_txt_path = os.path.join(output_dir, f"{model_name}_response.txt")

    with open(output_txt_path, "w") as f_out:
        for i, image_path in enumerate(image_list):
            try:
                print(f"[{i+1}/{len(image_list)}] {model_name} Processing: {image_path}")
                text = model.response(image_path, query)
                print(f"{model_name}:", text)

                f_out.write(f"Image: {image_path}\n")
                f_out.write(f"{model_name}: {text.strip()}\n")
                f_out.write("="*80 + "\n\n")
            except Exception as e:
                print(f"Error processing {image_path} with {model_name}: {e}")
                f_out.write(f"Image: {image_path}\n")
                f_out.write(f"Error: {e}\n")
                f_out.write("="*80 + "\n")

    del model
    gc.collect()
    print(f"Finished {model_name}\n{'#'*80}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Visual Prompt Evaluation using Vision-Language Models")

    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model responses')
    parser.add_argument('--query', type=str, required=False, default="Describe the main object in the scene that is most likely to influence the vehicle's next driving decision. You only need to describe the object in JSON format {'object': ,'describe:' }.", help='Textual query to ask each image')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Model name to run, e.g., gpt-4o')
    parser.add_argument('--stream', action='store_true', help='Enable stream mode (only used by certain models like Qwen-qvq)')
    
    return parser.parse_args()


def main():
    args = parse_args()

    image_list = [
        os.path.join(args.image_dir, fname)
        for fname in sorted(os.listdir(args.image_dir))
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Add more models here if needed
    models_info = [
        (Gpt_model, args.model, args.model),
    ]

    for model_class, model_name, model_path in models_info:
        process_model(model_class, model_name, model_path, args.output_dir, image_list, args.query, args.stream)


if __name__ == "__main__":
    main()
