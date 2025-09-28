import torch
import json
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
import os
import random
import numpy as np
import re
import csv
import argparse

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content

import torch.distributed as dist
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def build_message_with_images(
    prompt,
    images,
    max_pixels = 12_845_056,
):
    parts = prompt.split("<image>")
    if len(parts) - 1 != len(images):
        raise ValueError(f"Prompt has {len(parts)-1} <image> tokens but you passed {len(images)} images.")
    
    content = []
    for i, text_part in enumerate(parts):
        # 1) emit the text before the i‑th <image>
        if text_part:
            content.append({"type": "text", "text": text_part})
        # 2) if there's a corresponding image, emit it
        if i < len(images):
            img = images[i]
            content.append({
                "type": "image",
                "image": img,
                "max_pixels": max_pixels
            })
    return [{"role": "user", "content": content}]


def get_first_alphabetic_letter(input_string):
    # Find the first alphabetic letter in the string
    match = re.search(r'[A-Za-z]', input_string)
    if match:
        return match.group(0)
    else:
        return ''  # Return None if no alphabetic letter is found

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../eval_data", help='Path to eval data')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct", help='Path to model')
    parser.add_argument('--save_name', type=str, default="Qwen2.5-VL-7B-Instruct.json", help='Path to save the results')
    args = parser.parse_args()

    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"

    DATA_PATH = os.path.join(args.data_path, 'MMVP/Questions.csv')
    IMAGE_ROOT = os.path.join(args.data_path, 'MMVP/images')
    MODEL_PATH = args.model_path
    OUTPUT_PATH = os.path.join('./logs/mmvp/', args.save_name)

    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}, 
    )

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    data = []
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    for i in range(0, len(data), 2):
        data[i]['pair_index'] = i//2
        data[i+1]['pair_index'] = i//2

    random.seed(42)
    random.shuffle(data)

    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []

    if rank == 0:
        print('preprocess_data')
    for sample in tqdm(rank_data, disable=rank != 0):
        image = Image.open(os.path.join(IMAGE_ROOT,sample['Index']+'.jpg')).convert('RGB')
        question = '<image>\n' + sample['Question'] + "\nOptions: " + sample['Options'] + "\nAnswer with the option letter with parenthesis directly."

        message = build_message_with_images(question, [image])

        messages.append(message)

    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    if rank == 0:
        print('inference')
    # Process data
    for msg in tqdm(messages, disable=rank != 0):
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)]
        
        image_inputs, video_inputs = process_vision_info([msg])
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
            
        rank_outputs.extend(batch_output_text)

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)

    assert gathered_results[-1][-1][0] == len(data) - 1

    # The main process will collect all results
    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []
        correct_list = []

        for input_example, model_output in zip(data, all_outputs):
            answer = input_example['Correct Answer'][1]
            model_output = model_output.strip().strip('".').strip("'")
            model_output = model_output.split('</think>')[-1].strip()
            boxed_answer = extract_boxed_content(model_output)
            if boxed_answer != 'None':
                model_output = boxed_answer
            model_output = get_first_alphabetic_letter(model_output)

            correct = int(model_output.lower() == answer.lower())
            correct_list.append(correct)
            
            # Create a result dictionary for this example
            result = {
                'ground_truth': answer,
                'model_output': model_output,
                'correct': correct
            }
            final_output.append(result)

        # Calculate and print accuracy
        pair_accuracy_list = [correct_list[i] and correct_list[i+1] for i in range(0, len(correct_list), 2)]

        accuracy = np.mean(correct_list) * 100
        print(f"\nAccuracy: {accuracy:.2f}")

        pair_accuracy = np.mean(pair_accuracy_list) * 100
        print(f"\n Pair Accuracy: {pair_accuracy:.2f}")

        # Save results to a JSON file
        output_path = OUTPUT_PATH
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'pair_accuracy': pair_accuracy,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)

    # Synchronize all processes
    dist.barrier()