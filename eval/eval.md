## Evaluation Guidelines

We provide the evaluation scripts for the benchmarks used in our work (benchmarks not included can be directly evaluated using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)).

For evaluation, you need to first download and unzip some pre-processed data files from [here](https://huggingface.co/datasets/craigwu/visual_jigsaw_eval_data).

Run the following command to launch the evaluation for a certain benchmark:  
```
torchrun --nproc_per_node=$GPUS eval_XXX.py --data_path $EVAL_DATA_FOLDER --model_path $MODEL_PATH --save_name $SAVE_NAME
```  
For video benchmarks, you can also specify the maximum frames by setting `--max_frames`.

Below we provide detailed guidelines for each benchmark:

- MMVP  
  The eval script is `eval_mmvp.py`. You need to download the original data from [MMVP](https://huggingface.co/datasets/MMVP/MMVP) and make sure you have the following data structure.
  ```
  eval_data/
  └── MMVP/
       ├── images/
       └── Questions.csv
  ```

- HRBench  
  The eval script is `eval_hrbench.py`. You need to download the original data from [HRBench](https://huggingface.co/datasets/DreamMr/HR-Bench) and make sure you have the following data structure.
  ```
  eval_data/
  └── HR-Bench/
  ```

- Lisa-Grounding  
  The eval script is `eval_lisa_grounding.py`. Make sure you have the following data structure.
  ```
  eval_data/
  └── lisa_grounding/
       ├── test/
       └── lisa_test.json
  ```

- OVD-Eval  
  The eval script is `eval_ovd_eval.py`. You need to download the original data from [OVDEval](https://huggingface.co/datasets/omlab/OVDEval/tree/main) and make sure you have the following data structure.
  ```
  eval_data/
  └── OVDEval/
       ├── images/
       └── all_data.json
  ```

- VSR  
  The eval script is `eval_vsr.py`. You need to download the COCO 2017 images and make sure you have the following data structure.
  ```
  eval_data/
  ├── coco/
  │    ├── train2017/
  │    └── val2017/
  └── vsr/
       └── test.jsonl
  ```

- OmniSpatial  
  The eval script is `eval_omnispatial.py`. You need to download the original data from [OmniSpatial](https://huggingface.co/datasets/qizekun/OmniSpatial) and make sure you have the following data structure.
  ```
  eval_data/
  └── OmniSpatial/
       ├── OmniSpatial_1.5K/
       └── all_eval_data.json
  ```

- DA-2K  
  The eval script is `eval_da2k.py`. You need to download the original data from [DA-2K](https://huggingface.co/datasets/depth-anything/DA-2K) and make sure you have the following data structure.
  ```
  eval_data/
  └── DA-2K/
       ├── images/
       └── all_data.json
  ```

- Winoground  
  The eval script is `eval_winoground.py`. You need to download the original data from [winoground](https://huggingface.co/datasets/facebook/winoground) and make sure you have the following data structure.
  ```
  eval_data/
  └── winoground/
       ├── data/images/
       └── all_data.json
  ```

- SugarCrepe++  
  The eval script is `eval_sugarcrepe_pp.py`. You need to download the COCO 2017 images and make sure you have the following data structure.
  ```
  eval_data/
  ├── coco/
  │    ├── train2017/
  │    └── val2017/
  └── SugarCrepe_pp/
       └── all_data.json
  ```

- AoTBench  
  The eval script is `eval_aotbench.py`. You need to download the original data from [AoTBench](https://huggingface.co/datasets/sherryxzh/AoTBench) and make sure you have the following data structure.
  ```
  eval_data/
  └── AoTBench/
       ├── videos/
       └── valid_data.json
  ```

- TOMATO  
  The eval script is `eval_tomato.py`. You need to download the original data from [TOMATO](https://huggingface.co/datasets/yale-nlp/TOMATO) and make sure you have the following data structure.
  ```
  eval_data/
  └── TOMATO/
       ├── videos/
       └── all_data.json
  ```

- FAVOR-Bench  
  The eval script is `eval_favor_bench.py`. You need to download the original data from [FAVOR](https://huggingface.co/datasets/zl2048/FAVOR) and make sure you have the following data structure.
  ```
  eval_data/
  └── FAVOR/
       ├── videos/FAVOR-Bench/
       └── question_perspective.json
  ```

- TUNA-Bench  
  The eval script is `eval_tunabench.py`. You need to download the original data from [TUNA-Bench](https://huggingface.co/datasets/friedrichor/TUNA-Bench) and make sure you have the following data structure.
  ```
  eval_data/
  └── TUNA/
       ├── video/
       └── TUNA-MCQ.json
  ```

- TVBench  
  The eval script is `eval_tvbench.py`. You need to download the original data from [TVBench](https://huggingface.co/datasets/FunAILab/TVBench) and make sure you have the following data structure.
  ```
  eval_data/
  └── TVBench/
       ├── video/
       └── all_data.json
  ```

- MotionBench    
  The eval script is `eval_motionbench.py`. You need to download the original data from [MotionBench](https://huggingface.co/datasets/zai-org/MotionBench) and make sure you have the following data structure.
  ```
  eval_data/
  └── MotionBench/
       ├── MotionBench/
       │    ├── self-collected/
       │    └── public-dataset/
       └── motionbench_val.json
  ```

- LVBench   
  The eval script is `eval_lvbench.py`. You need to download the original data from [LVBench](https://huggingface.co/datasets/zai-org/LVBench) and make sure you have the following data structure.
  ```
  eval_data/
  └── LVBench/
       ├── all_videos/
       └── all_data.json
  ```

- Video-TT    
  The eval script is `eval_videott.py`. You need to download the original data from [video-tt](https://huggingface.co/datasets/lmms-lab/video-tt) and make sure you have the following data structure.
  ```
  eval_data/
  └── video-tt/
       ├── Benchmark-AllVideos-HQ-encoded/
       └── test-00000-of-00001.parquet
  ```

- CVBench    
  The eval script is `eval_CVBench.py`. You need to download the original data from [CVBench](https://huggingface.co/datasets/Dongyh35/CVBench) and make sure you have the following data structure.
  ```
  eval_data/
  └── CVBench/
       ├── videos/
       └── all_data.json
  ```

- SAT-Real  
  The eval script is `eval_sat.py`. You need to download the original data from [SAT](https://huggingface.co/datasets/array/SAT) and make sure you have the following data structure.
  ```
  eval_data/
  └── sat/
       └── SAT_test.parquet
  ```

- 3DSRBench  
  The eval script is `eval_3dsrbench.py`. You need to download the COCO 2017 images and make sure you have the following data structure.
  ```
  eval_data/
  ├── coco/
  │    ├── train2017/
  │    └── val2017/
  └── 3DSRBench/
       └── all_data.json
  ```
  
- ViewSpatial    
  The eval script is `eval_viewspatial.py`. You need to download the original data from [ViewSpatial-Bench](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) and make sure you have the following data structure.
  ```
  eval_data/
  └── ViewSpatial-Bench/
  ```

- All-Angles-Bench  
  The eval script is `eval_allangles.py`. You need to download the original data from [All-Angles-Bench](https://huggingface.co/datasets/ch-chenyu/All-Angles-Bench) and make sure you have the following data structure.
  ```
  eval_data/
  └── All-Angles-Bench/
       ├── ego_exo4d_scenes/
       ├── egohumans_scenes/
       └── data.json
  ```

