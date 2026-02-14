import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import logging
import math
import shutil
import json
import random
import time
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Union
from multiprocessing import Pool, cpu_count

# Third-party imports
import numpy as np
import pandas as pd
import cv2
import ffmpeg
from PIL import Image
from tqdm.auto import tqdm

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as TT
from torchvision import transforms
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode

# Hugging Face imports
import transformers
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from huggingface_hub import create_repo, upload_folder
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

# Diffusers imports
import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3D3BModel_sep,
    CogvideoXBranchModel,
    CogVideoXI2VTriInpaintPipeline_sep,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    clear_objs_and_retain_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
    load_image,
    load_video
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available

from mask_process import transform_video_masks

import warnings

warnings.filterwarnings('ignore')

if is_wandb_available():
    import wandb

    os.environ["WANDB_API_KEY"] = "INPUT_YOUR_API_HERE"
    os.environ["WANDB_MODE"] = "offline"

# Configure logging
logger = get_logger(__name__)

def calculate_max_ratio_torch(video_tensor, output_height=576):
    max_ratio = 0.0
    
    for frame in video_tensor:  # frame shape: [3, H, W]
        gray = 0.2989 * frame[0] + 0.5870 * frame[1] + 0.1140 * frame[2]
        
        binary = (gray < 251).float()
        
        nonzero = torch.nonzero(binary)
            
        min_y = nonzero[:, 0].min()
        max_y = nonzero[:, 0].max()
        object_height = (max_y - min_y).item()
        
        height_ratio = object_height / output_height
        
        if height_ratio > max_ratio:
            max_ratio = height_ratio
    
    return max_ratio


def apply_consistent_gamma(frames, gamma=None, gamma_range=(0.5, 2)):
    if gamma is None:
        gamma = random.uniform(*gamma_range)
    
    frames_normalized = frames.float() / 255.0
    
    corrected_frames = torch.pow(frames_normalized, gamma) * 255.0
    
    return corrected_frames.to(frames.dtype), gamma
    

def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm ** 0.5
    return norm


def concatenate_images_horizontally(images1, images2, images3, images4, output_type="np"):
    '''
    Concatenate three lists of images horizontally.
    Args:
        images1: List[Image.Image] or List[np.ndarray]
        images2: List[Image.Image] or List[np.ndarray]
        images3: List[Image.Image] or List[np.ndarray]
    Returns:
        List[Image.Image] or List[np.ndarray]
    '''
    concatenated_images = []
    for img1, img2, img3, img4 in zip(images1, images2, images3, images4):
        # Convert images to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        arr3 = np.array(img3)
        arr4 = np.array(img4)

        # Concatenate arrays horizontally
        concatenated_img = np.concatenate((arr1, arr2, arr3, arr4), axis=1)

        # Convert back to PIL Image
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        elif output_type == "np":
            pass
        else:
            raise NotImplementedError
        concatenated_images.append(concatenated_img)
    return concatenated_images


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for GenCompositor.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--videopainter_branch_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validating_steps",
        type=int,
        default=50,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        # default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--runs_name", type=str, default=None, help="Runs name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--cogvideox_branch_name_or_path",
        type=str,
        default=None,
        help="The path to a pre-trained CogVideoX branch model to use for training.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--meta_file_path",
        type=str,
        default=None,
        help="The path to meta data.",
    )
    parser.add_argument(
        "--val_meta_file_path",
        type=str,
        default=None,
        help="The path to meta data.",
    )
    parser.add_argument(
        "--corrupt_file_path",
        type=str,
        default=None,
        help="The path to corrupt data.",
    )
    parser.add_argument(
        "--random_mask",
        action="store_true",
        help=(
            "Training CogVideoX with random mask"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[480, 720],
        help=(
            "The resolution for input videos, all the videos in the train/validation dataset will be resized to this"
            " resolution. Provide two integers: height and width."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_text_seq_length",
        type=int,
        default=226,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # branch
    parser.add_argument(
        "--fgbranch_layer_num",
        type=int,
        default=4,
        help="Number of layers in the branch.",
    )
    parser.add_argument(
        "--add_first",
        action='store_true',
        help="Enable add_first feature. Default is False.",
    )
    parser.add_argument(
        "--mask_add",
        action='store_true',
        help="Enable mask_add feature. Default is False.",
    )
    parser.add_argument(
        "--mask_background",
        action='store_true',
        help="Enable mask_background feature. Default is False.",
    )
    parser.add_argument(
        "--wo_text",
        action='store_true',
        help="Enable wo_text feature. Default is False.",
    )
    parser.add_argument(
        "--inpainting_loss_weight",
        type=float,
        default=1.0,
        help="The weight of inpainting loss.",
    )
    parser.add_argument(
        "--mix_train_ratio",
        type=float,
        default=0.0,
        help="The ratio of mix training of images and videos.",
    )
    parser.add_argument(
        "--first_frame_gt",
        action='store_true',
        help="Enable first_frame_gt feature. Default is False.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )
    parser.add_argument(
        "--noised_image_dropout",
        type=float,
        default=0.05,
        help="Image condition dropout probability when finetuning image-to-video.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default="resize",
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'resize']",
    )
    parser.add_argument(
        "--mask_transform_prob",
        type=float,
        default=0.7,
        help="Probability of transforming each SAM mask",
    )
    parser.add_argument(
        "--p_brush",
        type=float,
        default=0.3,
        help="Probability of using brush mask",
    )
    parser.add_argument(
        "--p_rect",
        type=float,
        default=0.3,
        help="Probability of using rectangle mask",
    )
    parser.add_argument(
        "--p_ellipse",
        type=float,
        default=0.2,
        help="Probability of using ellipse mask",
    )
    parser.add_argument(
        "--p_circle",
        type=float,
        default=0.2,
        help="Probability of using circle mask",
    )
    parser.add_argument(
        "--p_random_brush",
        type=float,
        default=0.0,
        help="Probability of using random brush mask",
    )
    parser.add_argument(
        "--margin_ratio",
        type=float,
        default=0.1,
        help="Margin ratio for mask boundary perturbation",
    )
    parser.add_argument(
        "--shape_scale_min",
        type=float,
        default=1.1,
        help="Minimum scale ratio for shape transformation",
    )
    parser.add_argument(
        "--shape_scale_max",
        type=float,
        default=1.5,
        help="Maximum scale ratio for shape transformation",
    )
    return parser.parse_args()


class VideoInpaintingDataset(Dataset):
    def __init__(
            self,
            meta_file_path: Optional[str] = None,
            instance_data_root: Optional[str] = None,
            dataset_name: Optional[str] = None,
            dataset_config_name: Optional[str] = None,
            caption_column: str = "text",
            video_column: str = "video",
            height: int = 480,
            width: int = 720,
            fps: int = 8,
            max_num_frames: int = 49,
            skip_frames_start: int = 0,
            skip_frames_end: int = 0,
            cache_dir: Optional[str] = None,
            id_token: Optional[str] = None,
            is_train: bool = True,
            load_mode: str = "online",
    ) -> None:
        super().__init__()
        self.meta_file_path = Path(meta_file_path)
        self.meta_file_path_str = meta_file_path
        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.instance_data_root_str = instance_data_root if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""
        self.is_train = is_train
        self.load_mode = load_mode

        if dataset_name is not None:
            self.instance_metas = self._load_dataset_from_hub()
        else:
            self.instance_metas = self._load_dataset_from_local_path()  # dataset_name: None, 走这条路
        if load_mode == "online":
            self.instance_metas = self._preprocess_data_online()
        elif load_mode == "offline":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.instance_metas)

    def __getitem__(self, index):
        if self.load_mode == "online":
            video_name, start_frame, end_frame, fps, prompt = self.instance_metas[index]

            end_frame = 97
            mask_path = os.path.join(self.instance_data_root_str, "filtered_mask", video_name)
            masked_video_path = os.path.join(self.instance_data_root_str, "filtered_masked_video", video_name)
            # 随机决定是否使用交换路径 (20% 概率)
            use_swapped_paths = random.random() < 0.2 and self.is_train
            if use_swapped_paths:
                # 交换路径模式 (20% 概率)
                video_path = os.path.join(self.instance_data_root_str, "inpainted_sum", video_name)
                fg_video_path = os.path.join(self.instance_data_root_str, "fg", 'white_video.mp4')
            else:
                video_path = os.path.join(self.instance_data_root_str, "GTs", video_name)
                fg_video_path = os.path.join(self.instance_data_root_str, "fg", video_name)

            cap = cv2.VideoCapture(masked_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            cap = cv2.VideoCapture(video_path)
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            out, _ = (
                ffmpeg
                .input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True)
            )

            mask_out, _ = (
                ffmpeg
                .input(mask_path)
                .output('pipe:', format='rawvideo', pix_fmt='gray')  # 使用灰度格式
                .run(capture_stdout=True, capture_stderr=True)
            )

            masked_video_out, _ = (
                ffmpeg
                .input(masked_video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True)
            )

            fg_out, _ = (
                ffmpeg
                .input(fg_video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True)
            )
            # 从这里开始remove ------------------------------------------------------------------------
            # transform the bytes data to numpy array RGB->BGR
            orig_frames = np.frombuffer(out, np.uint8).reshape(
                [-1, orig_height, orig_width, 3]
            )
                        # 计算总帧数
            pixels_per_frame = orig_height * orig_width * 3
            n_frames = len(out) // pixels_per_frame  # 自动推断帧数
            # 创建目标数组存放上采样后的帧
            upsampled_frames = np.empty([n_frames, height, width, 3], dtype=np.uint8)
            # 逐帧上采样
            for i in range(n_frames):
                # 使用双线性插值上采样
                upsampled_frames[i] = cv2.resize(
                    orig_frames[i], 
                    (width, height),       # 目标尺寸 (width, height)
                    interpolation=cv2.INTER_LINEAR
                )
            frames = upsampled_frames[..., ::-1][start_frame:end_frame]
            # 从这里结束remove ------------------------------------------------------------------------
            # 注释原代码的下面这两行，需要恢复！！！！！！ -----------------------------------------------
            # # transform the bytes data to numpy array RGB->BGR
            # frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])[..., ::-1]
            # frames = frames[start_frame:end_frame]
            video = np.array(frames)
            # tmp_binary_masks = (all_masks == int(mask_id)).astype(np.uint8)
            # Align to the 8 fps
            video = video[::int(fps // self.fps)]
            # tmp_binary_masks = tmp_binary_masks[::int(fps//self.fps)]

            masks = np.frombuffer(mask_out, np.uint8).reshape([-1, height, width, 1])[..., ::-1]  # (300, 1080, 1920, 1)
            masks = masks[start_frame:end_frame]
            mask_video = np.array(masks)
            mask_video = mask_video[::int(fps // self.fps)]

            masked_frames = np.frombuffer(masked_video_out, np.uint8).reshape([-1, height, width, 3])[...,
                            ::-1]  # (300, 1080, 1920, 1)
            masked_frames = masked_frames[start_frame:end_frame]
            masked_video = np.array(masked_frames)
            masked_video = masked_video[::int(fps // self.fps)]

            fg_frames = np.frombuffer(fg_out, np.uint8).reshape([-1, 576, 576, 3])[..., ::-1]  # (300, 576, 576, 3)
            fg_frames = fg_frames[start_frame:end_frame]
            fg_video = np.array(fg_frames)
            fg_video = fg_video[::int(fps // self.fps)]

            return {
                "prompt": self.id_token + prompt,
                "video": video,
                "binary_masks": mask_video,
                "masked_video": masked_video,
                "fg_video": fg_video,
                "fg_path": fg_video_path,
                "use_swapped_paths": use_swapped_paths,
            }
        else:
            raise NotImplementedError

    def _load_dataset_from_hub(self):
        raise NotImplementedError

    def _load_dataset_from_local_path(self):
        '''
        read the meta file and corrupt file, and return the metas
        '''
        if not self.meta_file_path.exists():
            raise ValueError("Meta file does not exist")
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        metas = pd.read_csv(self.meta_file_path)
        metas = metas[metas['caption'].str.len() > 50]
        metas = metas[(metas['end_frame'] - metas['start_frame']) // (metas['fps'] // self.fps) >= self.max_num_frames]
        if self.is_train:
            metas = metas[:]
            metas = metas.values
        else:
            metas = metas[:]
            metas = metas.values
        return metas

    def _preprocess_data_online(self):
        return self.instance_metas


class MyWebDataset():
    def __init__(self, resolution, tokenizer, random_mask, max_num_frames, max_sequence_length,
                 proportion_empty_prompts, is_train=True, mask_background=False, mix_train_ratio=0.0,
                 first_frame_gt=False, video_reshape_mode="resize", random_flip: Optional[float] = None,
                 mask_transform_prob=0.7, p_brush=0.3, p_rect=0.3, p_ellipse=0.2, p_circle=0.2, p_random_brush=0.0,
                 margin_ratio=0.1, shape_scale_min=1.1, shape_scale_max=1.5):
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.random_mask = random_mask
        self.max_num_frames = max_num_frames
        self.val_max_num_frames = max_num_frames
        self.max_sequence_length = max_sequence_length
        self.proportion_empty_prompts = proportion_empty_prompts
        self.mask_background = mask_background
        self.is_train = is_train
        self.mix_train_ratio = mix_train_ratio
        self.first_frame_gt = first_frame_gt
        self.video_reshape_mode = video_reshape_mode

        self.mask_transform_prob = mask_transform_prob
        self.p_brush = p_brush
        self.p_rect = p_rect
        self.p_ellipse = p_ellipse
        self.p_circle = p_circle
        self.p_random_brush = p_random_brush
        self.margin_ratio = margin_ratio
        self.shape_scale_min = shape_scale_min
        self.shape_scale_max = shape_scale_max

        self.resolutions = (max_num_frames, resolution[0], resolution[1])

    def tokenize_captions(self, caption, is_train=True):
        if random.random() < self.proportion_empty_prompts:
            caption = ""
        elif isinstance(caption, str):
            caption = caption
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption = random.choice(caption) if is_train else caption[0]
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            caption, max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def _find_nearest_resolution(self, height, width):
        # nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        nearest_res = [self.resolutions[0], self.resolutions[1], self.resolutions[2]]
        return nearest_res[1], nearest_res[2]

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def __call__(self, examples):
        pixel_values = []
        conditioning_pixel_values = []
        masks = []
        input_ids = []
        fg_pixel_values = []
        # output_file_path = 'datasets.txt'
        if self.is_train:

            for example in examples:
                caption = example["prompt"]
                video = example["video"]  # frame, height, width, c
                masks_video = example["binary_masks"]
                masked_video = example["masked_video"]
                frame, height, width, c = video.shape

                fg_video = example["fg_video"]
                use_swapped_paths = example["use_swapped_paths"]

                if frame > self.max_num_frames:
                    begin_idx = random.randint(0, frame - self.max_num_frames)
                    end_idx = begin_idx + self.max_num_frames
                    video = video[begin_idx:end_idx]

                    masked_video = masked_video[begin_idx:end_idx]
                    fg_video = fg_video[begin_idx:end_idx]
                    masks_video = masks_video[begin_idx:end_idx]
                    frame = end_idx - begin_idx
                elif frame <= self.max_num_frames:
                    remainder = (3 + (frame % 4)) % 4
                    if remainder != 0:
                        video = video[:-remainder]
                        masked_video = masked_video[:-remainder]
                        fg_video = fg_video[:-remainder]
                        masks_video = masks_video[:-remainder]
                    frame = video.shape[0]

                # video:  (49, 1080, 1920, 3) | mask:  (49, 1080, 1920, 1)
                video = torch.from_numpy(video).permute(0, 3, 1, 2)
                masked_video = torch.from_numpy(masked_video).permute(0, 3, 1, 2)
                fg_video = torch.from_numpy(fg_video).permute(0, 3, 1, 2)
                masks_video = torch.from_numpy(masks_video).permute(0, 3, 1, 2)
                # video: [49, 3, 1080, 1920] | mask: [49, 1, 1080, 1920] | masked_video:[49, 3, 1080, 1920] | fg_video: [49, 3, 576, 576]

                nearest_res = self._find_nearest_resolution(video.shape[2], video.shape[3])
                if self.video_reshape_mode == 'center':
                    video_resized = self._resize_for_rectangle_crop(video, nearest_res)

                    masked_resized = self._resize_for_rectangle_crop(masked_video, nearest_res)
                elif self.video_reshape_mode == 'resize':
                    video_resized = torch.stack([resize(frame, nearest_res) for frame in video], dim=0)

                    masked_resized = torch.stack([resize(frame, nearest_res) for frame in masked_video], dim=0)

                    if not use_swapped_paths:
                        max_ratio = calculate_max_ratio_torch(fg_video, fg_video.shape[2])
                    else:
                        max_ratio = 1
                    if max_ratio > 0.5:
                        fg_resized = torch.stack([resize(frame, (nearest_res[0], nearest_res[0])) for frame in fg_video], dim=0)
                    else:
                        crop_total = fg_video.shape[2] - nearest_res[0]
                        crop_each_side = crop_total // 2
                        # 进行对称裁剪
                        fg_resized = fg_video[:, :, crop_each_side:fg_video.shape[2]-crop_each_side, :]
                    fg_resized, used_gamma = apply_consistent_gamma(fg_resized, gamma_range=(0.3, 1.9))

                    masks_resized = torch.stack([resize(frame, nearest_res) for frame in masks_video], dim=0)
                else:
                    raise NotImplementedError
                video = video_resized
                masked_video = masked_resized
                masks_video = masks_resized

                target_height, target_width = video.shape[2], video.shape[3]
                pad_width = target_width - fg_resized.shape[-1]  # 720 - 480 = 240,    720 - 576 = 144
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                # 2. 填充张量（填充值为 255）
                fg_video = torch.nn.functional.pad(
                    fg_resized,
                    (pad_left, pad_right, 0, 0),  # 仅填充宽度维度（左右），高度不填充
                    mode="constant",
                    value=255
                )

                video = video.permute(0, 2, 3, 1).numpy()
                masked_video = masked_video.permute(0, 2, 3, 1).numpy()
                fg_video = fg_video.permute(0, 2, 3, 1).numpy()
                masks_video = masks_video.permute(0, 2, 3, 1).numpy()
                for i in range(frame):
                    video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)  # (480, 720, 3)
                    masked_video[i] = cv2.cvtColor(masked_video[i], cv2.COLOR_BGR2RGB)
                    fg_video[i] = cv2.cvtColor(fg_video[i], cv2.COLOR_BGR2RGB)

                    _, binary_mask = cv2.threshold(masks_video[i], 128, 255, cv2.THRESH_BINARY)  # (480, 720)
                    masks_video[i] = binary_mask[..., None]

                video = (video.astype(np.float32) / 127.5) - 1.0
                masked_video = (masked_video.astype(np.float32) / 127.5) - 1.0
                fg_video = (fg_video.astype(np.float32) / 127.5) - 1.0
                masks_video = (masks_video.astype(np.float32) / 255)

                pixel_values.append(torch.tensor(video).permute(0, 3, 1, 2))
                conditioning_pixel_values.append(torch.tensor(masked_video).permute(0, 3, 1, 2))
                fg_pixel_values.append(torch.tensor(fg_video).permute(0, 3, 1, 2))
                masks.append(torch.tensor(masks_video).permute(0, 3, 1, 2))
                input_ids.append(self.tokenize_captions(caption)[0])

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            conditioning_pixel_values = torch.stack(conditioning_pixel_values)
            conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
            fg_pixel_values = torch.stack(fg_pixel_values)
            fg_pixel_values = fg_pixel_values.to(memory_format=torch.contiguous_format).float()

            masks = torch.stack(masks)
            masks = masks.to(memory_format=torch.contiguous_format).float()

            input_ids = torch.stack(input_ids)


        else:
            for example in examples:
                caption = example["prompt"]
                video = example["video"]  # frame, height, width, c
                mask = example["binary_masks"]
                masked_video = example["masked_video"]
                fg_video = example["fg_video"]

                frame, height, width, c = video.shape
                assert mask.shape[0] == frame, "mask shape is not equal to video shape"

                if frame > self.max_num_frames:
                    begin_idx = random.randint(0, frame - self.max_num_frames)
                    end_idx = begin_idx + self.max_num_frames
                    video = video[begin_idx:end_idx]

                    masked_video = masked_video[begin_idx:end_idx]
                    fg_video = fg_video[begin_idx:end_idx]
                    mask = mask[begin_idx:end_idx]
                    frame = end_idx - begin_idx
                elif frame <= self.max_num_frames:
                    remainder = (3 + (frame % 4)) % 4
                    if remainder != 0:
                        video = video[:-remainder]
                        masked_video = masked_video[:-remainder]
                        fg_video = fg_video[:-remainder]
                        mask = mask[:-remainder]
                    frame = video.shape[0]

                mask = np.repeat(mask, 3, axis=3)

                assert len(mask) == len(video), f"mask shape is not equal to video shape for {example['prompt']}"

                target_height, target_width = video.shape[1], video.shape[2]  # 1080, 1920
                fg_resized = np.zeros((video.shape[0], video.shape[1], video.shape[1], video.shape[3]),
                                      dtype=fg_video.dtype)
                for i in range(fg_video.shape[0]):
                    fg_resized[i] = cv2.resize(
                        fg_video[i],
                        (video.shape[1], video.shape[1]),
                        interpolation=cv2.INTER_CUBIC  # 也可以选 INTER_LINEAR 或 INTER_NEAREST
                    )
                fg_resized = torch.from_numpy(fg_resized).permute(0, 3, 1, 2)
                fg_resized, used_gamma = apply_consistent_gamma(fg_resized, gamma_range=(0.5, 1.7))
                fg_resized = fg_resized.permute(0, 2, 3, 1).numpy()
                
                pad_width = target_width - fg_resized.shape[2]  # 1920 - 1080 = 840
                pad_left = pad_width // 2  # 420

                # 2. 填充张量（填充值为 255）
                fg_video = np.full(
                    (fg_resized.shape[0], target_height, target_width, fg_resized.shape[3]),
                    255,  # 填充值（白色）
                    dtype=fg_resized.dtype
                )
                fg_video[:, :, pad_left: pad_left + fg_resized.shape[2], :] = fg_resized

                mask_ = [Image.fromarray(mask[i]) for i in range(frame)]
                video_ = [Image.fromarray(cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)) for i in range(frame)]
                masked_video_ = [Image.fromarray(cv2.cvtColor(masked_video[i], cv2.COLOR_BGR2RGB)) for i in
                                 range(frame)]
                fg_video_ = [Image.fromarray(cv2.cvtColor(fg_video[i], cv2.COLOR_BGR2RGB)) for i in range(frame)]

                pixel_values.append(video_)
                conditioning_pixel_values.append(masked_video_)
                masks.append(mask_)
                input_ids.append(caption)
                fg_pixel_values.append(fg_video_)

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "fg_pixel_values": fg_pixel_values,
            "masks": masks,
            "input_ids": input_ids,
        }


def log_validation(
        pipe,
        args,
        accelerator,
        pipeline_args,
        epoch,
        validating_step=0,
        is_final_validation: bool = False,
):
    logger.info(f"Running validation...")
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    pipeline_args["prompt"] = pipeline_args["prompt"][0]
    for _ in range(args.num_validation_videos):    
        video = pipe(**pipeline_args, num_inference_steps=10, generator=generator, output_type="np").frames[0]
        # Concatenate images horizontally
        original_video = pipe.video_processor.preprocess_video(pipeline_args['video'][0], height=video.shape[1],
                                                               width=video.shape[2])
        masks = pipe.masked_video_processor.preprocess_video(pipeline_args['mask_video'][0], height=video.shape[1],
                                                             width=video.shape[2])
        fg = pipe.video_processor.preprocess_video(pipeline_args['fg_video'][0], height=video.shape[1],
                                                             width=video.shape[2])                                                                                                 
        if args.mask_background:
            masked_video = original_video * (masks >= 0.5)
        else:
            masked_video = original_video * (masks < 0.5)
        original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
        masked_video = pipe.video_processor.postprocess_video(video=masked_video, output_type="np")[0]
        fg = pipe.video_processor.postprocess_video(video=fg, output_type="np")[0]
        video_ = concatenate_images_horizontally(
            original_video,
            masked_video,
            fg,
            video
        )
        videos.append(video_)   # stty echo

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else f"validation_{epoch + 1:05d}_{validating_step}"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir,
                                        f"{phase_name}_video_{i}_{pipeline_args['replace_gt']}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )
        # if tracker.name == "tensorboard":

    clear_objs_and_retain_memory([pipe])

    del pipe
    # torch.cuda.empty_cache()
    gc.collect()

    return videos


def _get_t5_prompt_embeds(
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if text_input_ids is None:
        batch_size = len(prompt)
    else:
        batch_size = text_input_ids.shape[0]

    if tokenizer is not None and text_input_ids is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
        tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False, token_ids=None
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            text_input_ids=token_ids,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                text_input_ids=token_ids,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
        height: int,
        width: int,
        num_frames: int,
        vae_scale_factor_spatial: int = 8,
        patch_size: int = 2,
        attention_head_dim: int = 64,
        device: Optional[torch.device] = None,
        base_height: int = 480,
        base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16


    transformer_vanilla = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )
    transformer = CogVideoXTransformer3D3BModel_sep.from_transformer(
        transformer=transformer_vanilla,
        load_weights_from_transformer=True,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    logger.info("Initializing CogVideoX branch weights from transformer")
    branch = CogvideoXBranchModel.from_pretrained(args.videopainter_branch_path, torch_dtype=load_dtype)

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional context encoder branch
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    branch.requires_grad_(False)

    # Total Model Parameters (excluding transformer_vanilla): 10,850,142,499
    # Transformer Parameters: 5,570,679,360
    # Text Encoder Parameters: 4,762,310,656
    # VAE Parameters: 215,583,907
    # Branch Parameters: 301,568,576


    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
                "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
                and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
                "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
                and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    branch.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            transformer.enable_xformers_memory_efficient_attention()
            branch.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        branch.enable_gradient_checkpointing()
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model: CogVideoXTransformer3D3BModel_sep
                    model = unwrap_model(model)
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()

            load_model = CogVideoXTransformer3D3BModel_sep.from_pretrained(input_dir, subfolder="transformer")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([model])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)
        cast_training_params([branch], dtype=torch.float32)

    # Optimization parameters
    branch_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    branch_parameters_with_lr = {"params": branch_parameters, "lr": args.learning_rate}
    params_to_optimize = [branch_parameters_with_lr]

    use_deepspeed_optimizer = (
            accelerator.state.deepspeed_plugin is not None
            and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
            accelerator.state.deepspeed_plugin is not None
            and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    logger.info(
        f"use_deepspeed_optimizer: {use_deepspeed_optimizer}, use_deepspeed_scheduler: {use_deepspeed_scheduler}")

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset = VideoInpaintingDataset(
        meta_file_path=args.meta_file_path,
        instance_data_root=args.instance_data_root,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column,
        video_column=args.video_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )
    validation_dataset = VideoInpaintingDataset(
        meta_file_path=args.val_meta_file_path,
        instance_data_root=args.instance_data_root,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column,
        video_column=args.video_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
        is_train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=MyWebDataset(
            resolution=args.resolution,
            tokenizer=tokenizer,
            random_mask=args.random_mask,
            max_num_frames=args.max_num_frames,
            max_sequence_length=args.max_text_seq_length,
            proportion_empty_prompts=args.proportion_empty_prompts,
            mask_background=args.mask_background,
            mix_train_ratio=args.mix_train_ratio,
            first_frame_gt=args.first_frame_gt,
            mask_transform_prob=args.mask_transform_prob,
            p_brush=args.p_brush,
            p_rect=args.p_rect,
            p_ellipse=args.p_ellipse,
            p_circle=args.p_circle,
            p_random_brush=args.p_random_brush,
            margin_ratio=args.margin_ratio,
            shape_scale_min=args.shape_scale_min,
            shape_scale_max=args.shape_scale_max,
            is_train=True,
        ),
        pin_memory=args.pin_memory,
        num_workers=args.dataloader_num_workers,
        persistent_workers=False
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=MyWebDataset(
            resolution=args.resolution,
            tokenizer=tokenizer,
            random_mask=args.random_mask,
            max_num_frames=args.max_num_frames,
            max_sequence_length=args.max_text_seq_length,
            proportion_empty_prompts=args.proportion_empty_prompts,
            mask_background=args.mask_background,
            first_frame_gt=args.first_frame_gt,
            mask_transform_prob=args.mask_transform_prob,
            p_brush=args.p_brush,
            p_rect=args.p_rect,
            p_ellipse=args.p_ellipse,
            p_circle=args.p_circle,
            p_random_brush=args.p_random_brush,
            margin_ratio=args.margin_ratio,
            shape_scale_min=args.shape_scale_min,
            shape_scale_max=args.shape_scale_max,
            is_train=False,
        ),
        pin_memory=args.pin_memory,
        num_workers=args.dataloader_num_workers,
        persistent_workers=False
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "GenCompositor"
        accelerator.init_trackers(
            project_name="wandb_usage",
            config=vars(args),
            init_kwargs={"wandb": {"name": args.runs_name}}
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_scale_factor_temporal = vae.config.temporal_compression_ratio

    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    empty_prompt_embeds = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        "",
        args.max_text_seq_length,
        accelerator.device,
        weight_dtype,
        requires_grad=False,
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):  # B, T, C, H, W
                if global_step < 1:
                    pixel_values = (batch["pixel_values"][0, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
                    conditioning_pixel_values = (batch["conditioning_pixel_values"][0, 0].permute(1, 2,
                                                                                                  0).cpu().numpy() + 1) * 127.5
                    pixel_values = pixel_values.clip(0, 255).astype(np.uint8)
                    conditioning_pixel_values = conditioning_pixel_values.clip(0, 255).astype(np.uint8)
                    pixel_values = cv2.cvtColor(pixel_values, cv2.COLOR_BGR2RGB)  # (480, 720, 3)
                    conditioning_pixel_values = cv2.cvtColor(conditioning_pixel_values,
                                                             cv2.COLOR_BGR2RGB)  # (480, 720, 3)

                    fg_values = (batch["fg_pixel_values"][0, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
                    fg_values = fg_values.clip(0, 255).astype(np.uint8)
                    fg_values = cv2.cvtColor(fg_values, cv2.COLOR_BGR2RGB)  # (480, 720, 3)

                    mask = batch["masks"][0, 0].permute(1, 2, 0).cpu().numpy() * 255
                    mask = np.repeat(mask, 3, axis=-1).astype(np.uint8)

                    combined_image = np.hstack((pixel_values, conditioning_pixel_values, mask, fg_values))
                    cv2.imwrite(f'{args.runs_name}_training_sample_1_{global_step}.png', combined_image)

                    if batch["masks"].shape[1] > 1:
                        pixel_values = (batch["pixel_values"][0, 1].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
                        conditioning_pixel_values = (batch["conditioning_pixel_values"][0, 1].permute(1, 2,
                                                                                                      0).cpu().numpy() + 1) * 127.5

                        pixel_values = pixel_values.clip(0, 255).astype(np.uint8)
                        conditioning_pixel_values = conditioning_pixel_values.clip(0, 255).astype(np.uint8)

                        pixel_values = cv2.cvtColor(pixel_values, cv2.COLOR_BGR2RGB)
                        conditioning_pixel_values = cv2.cvtColor(conditioning_pixel_values, cv2.COLOR_BGR2RGB)

                        fg_values = (batch["fg_pixel_values"][0, 1].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
                        fg_values = fg_values.clip(0, 255).astype(np.uint8)
                        fg_values = cv2.cvtColor(fg_values, cv2.COLOR_BGR2RGB)  # (480, 720, 3)

                        mask = batch["masks"][0, 1].permute(1, 2, 0).cpu().numpy() * 255
                        mask = np.repeat(mask, 3, axis=-1).astype(np.uint8)

                        combined_image = np.hstack((pixel_values, conditioning_pixel_values, mask, fg_values))
                        cv2.imwrite(f'{args.runs_name}_training_sample_2_{global_step}.png', combined_image)

                images = batch["conditioning_pixel_values"][:, :1, :, :, :].permute(0, 2, 1, 3, 4)  # [B, C, F, H, W], [1,3,1,h,w]
                image_noise_sigma = torch.normal(
                    mean=-3.0, std=0.5, size=(images.size(0),), device=accelerator.device, dtype=weight_dtype
                )
                image_noise_sigma = torch.exp(image_noise_sigma)
                noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                image_latents = vae.encode(
                    noisy_images.to(dtype=weight_dtype)).latent_dist.sample()  # [1, 16, 1, 60, 90]

                image_latents = image_latents.permute(0, 2, 1, 3, 4) * vae.config.scaling_factor  # [B, F, C, H, W]
                image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                # [1, 49, 3, 480, 720] --- [1, 16, 13, 60, 90], GT video:
                model_input = vae.encode(batch["pixel_values"].permute(0, 2, 1, 3, 4).to(
                    dtype=weight_dtype)).latent_dist.sample()  # source video

                model_input = model_input.permute(0, 2, 1, 3, 4) * vae.config.scaling_factor  # [B, F, C, H, W]
                model_input = model_input.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # [1, 49, 3, 480, 720] --- [1, 16, 13, 60, 90], foreground video:
                fg_input = vae.encode(batch["fg_pixel_values"].permute(0, 2, 1, 3, 4).to(
                    dtype=weight_dtype)).latent_dist.sample()  # source video

                fg_input = fg_input.permute(0, 2, 1, 3, 4) * vae.config.scaling_factor  # [B, F, C, H, W]
                fg_input = fg_input.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # [1, 49, 3, 480, 720] --- [1, 16, 13, 60, 90], mased video:
                conditioning_latents = vae.encode(batch["conditioning_pixel_values"].permute(0, 2, 1, 3, 4).to(
                    dtype=weight_dtype)).latent_dist.sample()  # masked video

                conditioning_latents = conditioning_latents.permute(0, 2, 1, 3, 4) * vae.config.scaling_factor  # [B, F, C, H, W]
                conditioning_latents = conditioning_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)


                padding_shape = (conditioning_latents.shape[0], conditioning_latents.shape[1] - 1,
                                 *conditioning_latents.shape[2:])  # [1, 12, 16, 60, 90]
                latent_padding = image_latents.new_zeros(padding_shape)  # [1, 1, 16, 60, 90]
                image_latents = torch.cat([image_latents, latent_padding], dim=1)  # [1, 13, 16, 60, 90]

                if random.random() < args.noised_image_dropout:
                    image_latents = torch.zeros_like(image_latents)

                # torch.cuda.empty_cache()

                masks = batch["masks"].permute(0, 2, 1, 3, 4)  # (1,1,49,480,720)
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=(
                        (masks.shape[-3] - 1) // vae_scale_factor_temporal + 1,
                        args.height // vae_scale_factor_spatial,
                        args.width // vae_scale_factor_spatial
                    )
                ).permute(0, 2, 1, 3, 4).to(dtype=weight_dtype)
                masked_latents = conditioning_latents
                conditioning_latents = torch.concat([conditioning_latents, masks], -3).to(dtype=weight_dtype)
                prompts = batch["prompts"] if batch['input_ids'] is None else 'xx'

                # encode prompts
                prompt_embeds = compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    prompts,
                    args.max_text_seq_length,
                    accelerator.device,
                    weight_dtype,
                    requires_grad=False,
                    token_ids=batch["input_ids"],
                )
                # Sample noise that will be added to the latents
                noise = torch.randn_like(model_input).to(dtype=weight_dtype)
                batch_size, num_frames, num_channels, height, width = model_input.shape
                # empty prompt
                empty_prompt_embeds = empty_prompt_embeds.repeat(batch_size, 1, 1)  # [1, 226, 4096]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )
                sep_image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height * 2,
                        width=args.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=480*2,
                        # base_width=720*2,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(model_input, noise, timesteps)

                noisy_model_input = torch.cat([noisy_video_latents, masked_latents, fg_input], dim=2)  # [1, 13, 32 + 16, 60, 90]

                branch_block_samples = branch(
                    hidden_states=noisy_video_latents,
                    encoder_hidden_states=prompt_embeds,
                    branch_cond=conditioning_latents,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    mask_add=args.mask_add,
                    wo_text=args.wo_text,
                    return_dict=False,
                )[0]
                branch_block_samples = [block_sample.to(dtype=weight_dtype) for block_sample in branch_block_samples]  # 2个

                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=sep_image_rotary_emb,
                    return_dict=False,
                    branch_block_samples=branch_block_samples,
                    branch_block_masks=masks if args.mask_add else None,
                    add_first=args.add_first,
                )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input

                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)  # 直接看整体
                loss = loss.mean()
                inpainting_loss = torch.mean(
                    (weights * (model_pred * masks - target * masks) ** 2).reshape(batch_size, -1), dim=1)  # 看补进去的东西
                inpainting_loss = inpainting_loss.mean()
                loss = loss + args.inpainting_loss_weight * inpainting_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    gradient_norm_before_clip = get_gradient_norm(params_to_clip)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(params_to_clip)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # torch.cuda.empty_cache()

                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "inpainting_loss": inpainting_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.distributed_type != DistributedType.DEEPSPEED:
                logs.update(
                    {
                        "gradient_norm_before_clip": gradient_norm_before_clip,
                        "gradient_norm_after_clip": gradient_norm_after_clip,
                    }
                )
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if (global_step % args.validating_steps == 0) or (global_step == 1):

                pipe = CogVideoXI2VTriInpaintPipeline_sep.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    branch=unwrap_model(branch),
                    scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                vali_num = 0
                for step, batch in enumerate(validation_dataloader):
                    pipeline_args = {
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "height": args.height,
                        "width": args.width,
                        "video": batch["pixel_values"],
                        "prompt": batch["input_ids"],
                        "mask_video": batch["masks"],
                        "masked_video": batch["conditioning_pixel_values"],
                        "fg_video": batch["fg_pixel_values"],
                        "num_frames": 49,
                        "strength": 1.0,
                        "mask_background": args.mask_background,
                        "add_first": args.add_first,
                        "wo_text": args.wo_text,
                        "mask_add": args.mask_add,
                        "replace_gt": (vali_num + 1) % 2 == 1,
                    }
                    validation_outputs = log_validation(
                        pipe=pipe,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                        validating_step=global_step,
                    )
                    del batch
                    # torch.cuda.empty_cache()
                    gc.collect()
                    vali_num += 1
                    if vali_num >= 2:
                        break
        if ((epoch + 1) % args.validation_epochs == 0) or (epoch == 0):
            try:
                pipe = CogVideoXI2VTriInpaintPipeline_sep.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    branch=unwrap_model(branch),
                    scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )

                for step, batch in enumerate(validation_dataloader):
                    if step < 1:
                        try:
                            pipeline_args = {
                                "guidance_scale": args.guidance_scale,
                                "use_dynamic_cfg": args.use_dynamic_cfg,
                                "height": args.height,
                                "width": args.width,
                                "video": batch["pixel_values"],
                                "prompt": batch["input_ids"],
                                "mask_video": batch["masks"],
                                "masked_video": batch["conditioning_pixel_values"],
                                "fg_video": batch["fg_pixel_values"],
                                "num_frames": 49,
                                "strength": 1.0,
                                "mask_background": args.mask_background,
                                "add_first": args.add_first,
                                "mask_add": args.mask_add,
                                "replace_gt": (vali_num + 1) % 2 == 1,
                            }
                            validation_outputs = log_validation(
                                pipe=pipe,
                                args=args,
                                accelerator=accelerator,
                                pipeline_args=pipeline_args,
                                epoch=epoch,
                                validating_step=global_step,
                            )
                        except Exception as e:
                            logger.error(f"Error during validation step: {e}")
                            continue

                        del batch
                        # torch.cuda.empty_cache()
                        gc.collect()

                    if step >= 1:
                        break

                del pipe
                # torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logger.error(f"Error during validation: {e}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)


