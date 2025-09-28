import torch
import json
from tqdm import tqdm
import base64
import io
from PIL import Image
import os
import random
import numpy as np
import re
import argparse
from collections import defaultdict

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

def extract_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        return content_answer
    return content

def extract_bbox_answer(content):
    content = extract_answer(content)
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    # bbox_pattern = r'\[(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\]'
    bbox_match = re.search(bbox_pattern, content)

    if bbox_match:
        bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
        return bbox
    return [0, 0, 0, 0]

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height
    return bbox

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

    DATA_PATH = os.path.join(args.data_path, 'lisa_grounding', 'lisa_test.json')
    IMAGE_ROOT = os.path.join(args.data_path, 'lisa_grounding')
    MODEL_PATH = args.model_path
    OUTPUT_PATH = os.path.join('./logs/lisa_grounding/', args.save_name)

    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}, 
    )

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    data = json.load(open(DATA_PATH, "r"))

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
        image = Image.open(os.path.join(IMAGE_ROOT,sample['image'])).convert('RGB')
        question = "<image>"+sample['problem'] + " Please provide the bounding box coordinate in JSON format."


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

        batch_output = []
        for i, output_text in enumerate(batch_output_text):
            input_height = int(inputs['image_grid_thw'][i][1]*14)
            input_width = int(inputs['image_grid_thw'][i][2]*14)
            image = msg[0]['content'][0]['image']
            image_width, image_height = image.size
            batch_output.append((output_text, input_height, input_width, image_height, image_width))
            
        rank_outputs.extend(batch_output)

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
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output, input_height, input_width, image_height, image_width = model_output
            ground_truth = input_example['solution']
            original_output = original_output.strip().strip('".').strip("'")
            original_output = original_output.split('</think>')[-1].strip()
            boxed_answer = extract_boxed_content(original_output)
            if boxed_answer != 'None':
                original_output = boxed_answer

            model_answer = extract_bbox_answer(original_output)
            resized_model_answer = resize_bbox(model_answer, input_height, input_width, image_height, image_width)

            correct = 0
            if model_answer is not None:
                if iou(resized_model_answer, ground_truth) > 0.5:
                    correct = 1
            correct_number += correct
            
            # Create a result dictionary for this example
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': resized_model_answer,
                'correct': correct
            }
            final_output.append(result)

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy: {accuracy:.2f}%")

        # Save results to a JSON file
        # output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
        output_path = OUTPUT_PATH
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)
        
    # Synchronize all processes
    dist.barrier()