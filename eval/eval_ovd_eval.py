import torch
import json
from tqdm import tqdm
from collections import defaultdict
import base64
from io import BytesIO
from PIL import Image
import os
import random
import numpy as np
import re
import argparse
import math
import collections
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 12845056
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def mapped_back_bbox(original_size, resized_size, bbox):
    o_w, o_h = original_size
    n_w, n_h = resized_size
    mapped_bbox = [0]*4
    mapped_bbox[0] = int(bbox[0] * n_w/o_w)
    mapped_bbox[1] = int(bbox[1] * n_h/o_h)
    mapped_bbox[2] = int(bbox[2] * n_w/o_w)
    mapped_bbox[3] = int(bbox[3] * n_h/o_h)

    mapped_bbox[0] = min(o_w-1, max(mapped_bbox[0], 0))
    mapped_bbox[1] = min(o_h-1, max(mapped_bbox[1], 0))
    mapped_bbox[2] = min(o_w-1, max(mapped_bbox[2], 0))
    mapped_bbox[3] = min(o_h-1, max(mapped_bbox[3], 0))

    return mapped_bbox


def extract_bboxes(s: str):
    """
    Extract all bounding boxes in the format [x1, y1, x2, y2] from the string.

    Args:
        s (str): Input string containing bbox-like patterns.

    Returns:
        list[list[int]]: List of bounding boxes, or [[0, 0, 0, 0]] if none found.
    """
    pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
    matches = re.findall(pattern, s)
    
    if not matches:
        return [[0, 0, 0, 0]]

    return [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in matches]

def calculate_iou(box1, boxes2):
    """
    Calculate the intersection ratio (IoU)
    """
    x1, y1, w1, h1 = box1
    x2 = boxes2[:, 0]
    y2 = boxes2[:, 1]
    w2 = boxes2[:, 2]
    h2 = boxes2[:, 3]

    xmin = np.maximum(x1, x2)
    ymin = np.maximum(y1, y2)
    xmax = np.minimum(x1 + w1, x2 + w2)
    ymax = np.minimum(y1 + h1, y2 + h2)

    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    union = w1 * h1 + w2 * h2 - intersection

    # Handle cases where the denominator is zero
    iou = np.where(union == 0, 0, intersection / union)

    return iou

def nms_threaded(gt, prediction, iou_threshold, num_threads):
    gt_boxes = np.array([box['bbox'] for box in gt])

    boxes = np.array([box['bbox'] for box in prediction])
    scores = np.array([box['score'] for box in prediction])

    keep_list = []
    remove_list = []
    for idx, i in enumerate(gt):
        gt_box = gt_boxes[idx]
        iou = calculate_iou(gt_box, boxes)
        indices =  np.where(iou > iou_threshold)[0].tolist()
        if indices:
            match_scores = scores[indices]
            sorted_indices = np.argsort(match_scores)[::-1]

            final_indices = []
            for sort_idx in sorted_indices.tolist():
                final_indices.append(indices[sort_idx])

            if final_indices:
                keep_list.append(final_indices[0])
                remove_list += final_indices[1:]

    final_remove_list = []
    for i in remove_list:
        if i not in keep_list:
            final_remove_list.append(i)

    selected_boxes = []
    for idx, i in enumerate(prediction):
        if idx not in final_remove_list:
            selected_boxes.append(i)

    return selected_boxes

def cal_map(gt_data, pred_data):
    
    anno = COCO(gt_data)
    pred = anno.loadRes(pred_data)
    eval = COCOeval(anno, pred, 'bbox')

    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    mAP = eval.stats[0]  # AP@[0.50:0.95]
    return mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../eval_data", help='Path to eval data')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct", help='Path to model')
    parser.add_argument('--save_name', type=str, default="Qwen2.5-VL-7B-Instruct.json", help='Path to save the results')
    args = parser.parse_args()

    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"

    DATA_PATH = os.path.join(args.data_path, 'OVDEval/all_data.json')
    DATA_ROOT = os.path.join(args.data_path, 'OVDEval')
    IMAGE_ROOT = os.path.join(args.data_path, 'OVDEval/images')

    MODEL_PATH = args.model_path
    OUTPUT_PATH = os.path.join('./logs/ovd_eval/', args.save_name)
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}, 
    )

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    data = json.load(open(DATA_PATH))
    for i, sample in enumerate(data):
        sample['index'] = i

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
        task = sample['task']
        image_info = sample['image_info']
        category_info = sample['category_info']
        image = Image.open(os.path.join(IMAGE_ROOT, task, image_info['file_name'])).convert('RGB')

        question = "<image>Detect all instances of the following target: {}.".format(category_info['name'])
        question += " Output the detected bbox coordinates in JSON format."
        
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

        for input_example, model_output in zip(data, all_outputs):
            model_output = model_output.split('</think>')[-1].strip()
            boxed_answer = extract_boxed_content(model_output)
            if boxed_answer != 'None':
                model_output = boxed_answer
            input_example['model_output'] = model_output
            final_output.append(input_example)


        # calculate metrics
        avg_map = []
        task2map = dict()
        for task in ['celebrity', 'color', 'landmark', 'logo', 'material', 'negation', 'position', 'relationship']:
        # for task in ['color']:
            gt_path = os.path.join(DATA_ROOT, 'annotations', f'{task}.json')
            pred_results_raw = [data for data in final_output if data['task'] == task]

            before_nms_results = []
            for pred_data in pred_results_raw:
                image_info = pred_data['image_info']
                category_info = pred_data['category_info']
                model_output = pred_data['model_output']
                pred_bbox_list = extract_bboxes(model_output)
                original_image_width = image_info['width']
                original_image_height = image_info['height']
                resized_image_height, resized_image_width = smart_resize(original_image_height, original_image_width)
                pred_bbox_list = [mapped_back_bbox([original_image_width, original_image_height], [resized_image_width, resized_image_height], bbox) for bbox in pred_bbox_list]
                pred_bbox_list = [[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] for bbox in pred_bbox_list]

                for bbox in pred_bbox_list:
                    before_nms_results.append({'image_id':image_info['id'], 'score':1, 'category_id': category_info['id'], 'bbox':bbox})

            iou_threshold = 0.5
            num_threads = 16
            
            # load annotation data
            gt_data = json.load(open(gt_path))
            
            # nms with gt
            gt_data_ann = gt_data['annotations']
            
            image_dict = collections.defaultdict(list)
            gt_dict = collections.defaultdict(list)

            for i in before_nms_results:
                image_dict[i["image_id"]].append(i)

            for i in gt_data_ann:
                gt_dict[i["image_id"]].append(i)
            
            after_nms_results = []
            # Call multithreaded parallelized NMS functions
            for img, preds in image_dict.items():
                gts = gt_dict[img]
                selected_boxes = nms_threaded(gts, preds, iou_threshold, num_threads)
                after_nms_results += selected_boxes
        
            mAP = cal_map(gt_path, after_nms_results)
            task2map[task] = mAP
            avg_map.append(mAP)

        for task, mAP in task2map.items():
            print('Task: {}, mAP: {}'.format(task, mAP))
        task2map['avg'] = np.mean(avg_map)
        print('Avg mAP:', np.mean(avg_map))

        # Save results to a JSON file
        output_path = OUTPUT_PATH
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'acc': task2map
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)

    # Synchronize all processes
    dist.barrier()