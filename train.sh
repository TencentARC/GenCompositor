export MODEL_PATH="../ckpts/CogVideoX-5b-I2V"
export BRANCH_PATH="../ckpts/branch"
export CACHE_PATH="~/.cache"
export DATASET_PATH="../VideoComp_demo"
export PROJECT_NAME="compositing"
export RUNS_NAME="GenCompositor_sep"
export OUTPUT_PATH="./${PROJECT_NAME}/${RUNS_NAME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --config_file accelerate_config_machine_single_ds.yaml  --machine_rank 0 \
  train_cogvideox_compositing_sep.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --videopainter_branch_path $BRANCH_PATH \
  --cache_dir $CACHE_PATH \
  --meta_file_path ./train_demo.csv \
  --val_meta_file_path ./val_demo.csv \
  --instance_data_root $DATASET_PATH \
  --dataloader_num_workers 1 \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_reshape_mode "resize" \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --max_text_seq_length 226 \
  --fgbranch_layer_num 1 \
  --train_batch_size 1 \
  --num_train_epochs 10 \
  --checkpointing_steps 1024 \
  --validating_steps 256 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 1000 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --noised_image_dropout 0.05 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --tracker_name $PROJECT_NAME \
  --runs_name $RUNS_NAME \
  --inpainting_loss_weight 5.0 \
  --mix_train_ratio 0 \
  --first_frame_gt \
  --mask_add \
  --mask_transform_prob 0.3 \
  --p_brush 0.4 \
  --p_rect 0.1 \
  --p_ellipse 0.1 \
  --p_circle 0.1 \
  --p_random_brush 0.3