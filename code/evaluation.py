import argparse
import os
import re
import json
import time
import openai
from openai.error import ServiceUnavailableError, Timeout, APIError

def extract_single_tag_outputs(text, tag):
    entries = re.split(r'=+', text)
    outputs = []
    for entry in entries:
        match = re.search(rf'{tag}:\s*(.*)', entry, re.DOTALL)
        if match:
            content = match.group(1).strip()
            code_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            if code_match:
                extracted = code_match.group(1).strip()
            else:
                extracted = content
            outputs.append(extracted)
    return outputs


class GPTScorer:
    def __init__(self, model: str = "gpt-4o", api_key: str = "", api_base: str = ""):
        self.model = model
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base

    def compute_similarity(self, input_text: str, reference_text: str, max_retries: int = 5) -> float:
        prompt = f"""Rate the semantic similarity between the following two texts on a scale from 0 to 1.

**Criteria for similarity measurement:**
1. **Main Subject Consistency:** If both descriptions refer to the same key subject or object (e.g., a person, food, an event), they should receive a higher similarity score.
2. **Relevant Description**: If the descriptions are related to the same context or topic, they should also contribute to a higher similarity score.
3. **Ignore Fine-Grained Details:** Do not penalize differences in **phrasing, sentence structure, or minor variations in detail**. Focus on **whether both descriptions fundamentally describe the same thing.**
4. **Partial Matches:** If one description contains extra information but does not contradict the other, they should still have a high similarity score.
5. **Similarity Score Range:** 
    - **1.0**: Nearly identical in meaning.
    - **0.8-0.9**: Same subject, with highly related descriptions.
    - **0.7-0.8**: Same subject, core meaning aligned, even if some details differ.
    - **0.5-0.7**: Same subject but different perspectives or missing details.
    - **0.3-0.5**: Related but not highly similar.
    - **0.0-0.2**: Completely different subjects or unrelated meanings.

Text 1: {input_text}
Text 2: "{reference_text}"

Output only a single number between 0 and 1. Do not include any explanation or additional text."""

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                score = response['choices'][0]['message']['content'].strip()
                return min(1.0, max(0.0, float(score)))
            except (ServiceUnavailableError, Timeout, APIError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}] GPT API failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"[Fatal] Unexpected error: {e}")
                break

        print("[Error] Failed to compute similarity after retries.")
        return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT outputs with semantic similarity scoring.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the GPT output .txt file")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model tag in the file (e.g., GPT-4o)")
    parser.add_argument("--reference_text", type=str, default="A pedestrian crossing sign is visible", help="Reference text to compare against")
    parser.add_argument("--start", type=int, default=0, help="Start index for evaluation")
    parser.add_argument("--end", type=int, default=100, help="End index for evaluation")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--api_base", type=str, default=None, help="Optional custom OpenAI API base")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.file_path, "r") as f:
        text_data = f.read()

    gpt_outputs = extract_single_tag_outputs(text_data, args.model_name)
    print(f" Extracted {len(gpt_outputs)} outputs for tag: {args.model_name}")

    scorer = GPTScorer(model=args.model_name, api_key=args.api_key, api_base=args.api_base)

    sim_list = []
    acc = 0

    start, end = args.start, min(args.end, len(gpt_outputs))
    for i in range(start, end):
        input_text = gpt_outputs[i]
        score = scorer.compute_similarity(input_text, reference_text=args.reference_text)
        sim_list.append(score)
        print(f"{i:03d} → {score:.4f}")
        if score > 0.5:
            acc += 1

    if sim_list:
        sim = sum(sim_list) / len(sim_list)
        print(f"\n Total passed: {acc}/{len(sim_list)} | Mean Similarity: {sim:.4f}")
    else:
        print("No valid outputs found or processed.")


if __name__ == "__main__":
    main()
