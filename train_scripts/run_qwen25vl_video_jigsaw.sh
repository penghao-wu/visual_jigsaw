set -x

WORLD_SIZE=1

export WANDB_API_KEY=TODO
export PROJECT_NAME="visual_jigsaw"
export EXPERIMENT_NAME="video_jigsaw_qwen25_7b"

SAVE_CHECKPOINT_DIR=./verl_checkpoints

mkdir -p ${SAVE_CHECKPOINT_DIR}
mkdir -p ${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME}

VISUAL_DATASET_TRAIN=TODO
VISUAL_DATASET_TEST=TODO
DATA_FOLDER=TODO

REF_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=32 \
    algorithm.adv_estimator=grpo \
    data.train_files=[${VISUAL_DATASET_TRAIN}] \
    data.val_files=[${VISUAL_DATASET_TEST}] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    +data.multimodal_folder=${DATA_FOLDER} \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=4000 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME} \
    trainer.total_epochs=32 2>&1 | tee ${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME}/logs.log
