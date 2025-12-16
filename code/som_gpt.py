import openai
import base64
import os
import re
import json
import argparse


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class Gpt_model:
    def __init__(self, model_name, api_key, api_base=None):
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base
        self.model_name = model_name

    def response(self, image_path_1, image_path_2, query):
        base64_image_1 = encode_image(image_path_1)
        base64_image_2 = encode_image(image_path_2)

        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image_1}"}},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image_2}"}}
            ]
        }]

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=message,
            temperature=1,
            max_tokens=512
        )

        return response['choices'][0]['message']['content']


def get_images_from_folder(folder_path):
    return sorted([
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-4o Label Region Extractor")
    parser.add_argument('--original_folder', type=str, required=True, help='Folder containing original images')
    parser.add_argument('--sam_folder', type=str, required=True, help='Folder containing SAM-labeled images')
    parser.add_argument('--label_folder', type=str, required=True, help='Folder containing label coordinate txts')
    parser.add_argument('--output_path', type=str, default='./coords.txt', help='Output path for center coordinates')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--api_base', type=str, default=None, help='Custom OpenAI API base (optional)')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name (default: gpt-4o)')
    return parser.parse_args()


def main():
    args = parse_args()

    text = """You will receive two input images:  
    The first image is the original scene;  
    The second image is the semantic segmentation map (SOM image) of the original scene, where each identifiable region has a unique numeric label displayed at the center of the mask.  

    Please analyze the images according to the following requirements and output the region ID that best meets the criteria:  

    - First, identify areas in the original image that are suitable for pasting or painting artistic advertisements.  
    - Then, locate the corresponding region in the SOM image and find its label number.  
    - The selected region must have a clearly visible numeric label in the segmentation image.  
    - It is allowed to select the surface of vehicles as the painting area, especially those parked near the edge.  
    - The selected region should be prominent in the image, flat, and easy to paint on.  
    - Prioritize objects with a frontal or side-on view.  
    - The object must not be occluded — the painting surface should be clearly visible and unobstructed.  
    - If an entity (such as a vehicle) consists of multiple labeled regions, choose the region with the largest area as the representative center label.  
    - The selected object should occupy a relatively large area in the current camera view, ensuring it is clearly visible and impactful for advertisement placement.
    - Prefer selecting large-looking buildings, vehicles, or other structures that appear visually dominant from the current camera perspective.

    Only select the single best region.  
    Return format: {"label": <center label>} — output only this JSON object."""

    gpt = Gpt_model(args.model, args.api_key, args.api_base)

    images1 = get_images_from_folder(args.original_folder)
    images2 = get_images_from_folder(args.sam_folder)
    centers = []

    for i, image_path in enumerate(images1):
        raw_output = gpt.response(images1[i], images2[i], text)
        match = re.search(r'\{.*?\}', raw_output)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                label_index = data.get("label")

                if isinstance(label_index, int):
                    image_filename = os.path.basename(image_path)
                    txt_filename = os.path.splitext(image_filename)[0] + ".txt"
                    txt_path = os.path.join(args.label_folder, txt_filename)

                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as f:
                            lines = f.readlines()

                        if 0 <= label_index < len(lines):
                            coord_line = lines[label_index - 1].strip()
                            x_str, y_str = coord_line.split(',')
                            x = float(x_str.strip())
                            y = float(y_str.strip())
                            centers.append((x, y))
                            print(f"{image_filename} → Label {label_index} → Coord: ({x}, {y})")
                        else:
                            print(f"[Warning] Label index {label_index} out of bounds in {txt_path}")
                    else:
                        print(f"[Error] Label file not found: {txt_path}")

            except json.JSONDecodeError:
                print("[Error] Invalid JSON format.")
        else:
            print("[Error] No JSON object found in response.")

    with open(args.output_path, 'w') as f:
        for x, y in centers:
            f.write(f"{x:.4f}, {y:.4f}\n")

    print(f" Saved {len(centers)} centers to {args.output_path}")


if __name__ == "__main__":
    main()
