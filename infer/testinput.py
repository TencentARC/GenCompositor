import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from typing import Literal
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers import (
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3D3BModel_sep,
    CogVideoXI2VTriInpaintPipeline_sep,
)
import cv2
from accelerate import Accelerator
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from io import BytesIO
import base64
from torchvision.transforms import ToTensor
import torch.nn as nn
from diffusers.utils.torch_utils import is_compiled_module
import random
from pathlib import Path


    
def quick_freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model



def apply_consistent_gamma(frames, gamma=None, gamma_range=(0.5, 2)):
    """
    对视频的所有帧应用相同的伽马校正（同一视频内伽马值固定）
    
    参数:
        frames: torch.Tensor, 形状为 (T, C, H, W) 的视频帧
        gamma: 如果为None，则随机生成；否则使用指定的gamma值
        gamma_range: 伽马值的随机范围 (min, max)
    
    返回:
        校正后的视频帧 (同输入形状), 使用的gamma值
    """
    # 生成或使用指定的gamma值
    if gamma is None:
        gamma = random.uniform(*gamma_range)
    
    # 归一化到[0,1]范围
    frames_normalized = frames.float() / 255.0
    
    # 应用相同的伽马校正到所有帧
    corrected_frames = torch.pow(frames_normalized, gamma) * 255.0
    
    return corrected_frames.to(frames.dtype), gamma




def get_gaussian_kernel(kernel_size, sigma, channels):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
        (2 * variance)
    )

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def _visualize_video(pipe, mask_background, original_video, video, masks, fg_video):

    original_video = pipe.video_processor.preprocess_video(original_video, height=video.shape[1], width=video.shape[2])
    fg_video = pipe.video_processor.preprocess_video(fg_video, height=video.shape[1], width=video.shape[2])
    masks = pipe.masked_video_processor.preprocess_video(masks, height=video.shape[1], width=video.shape[2])

    torch.cuda.empty_cache()

    if mask_background:
        masked_video = original_video * (masks >= 0.5)
    else:
        masked_video = original_video * (masks < 0.5)
    
    original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
    fg_video = pipe.video_processor.postprocess_video(video=fg_video, output_type="np")[0]
    masked_video = pipe.video_processor.postprocess_video(video=masked_video, output_type="np")[0]
    
    torch.cuda.empty_cache()
    
    masks = masks.squeeze(0).squeeze(0).numpy()
    masks = masks[..., np.newaxis].repeat(3, axis=-1)

    # video_ = concatenate_images_horizontally(
    #     [original_video, masked_video, fg_video, video],
    # )
    video_ = concatenate_images_horizontally(
        [original_video, masks, fg_video, video],
    )
    final_video = concatenate_images_horizontally(
        [video],
    )
    return video_, final_video

def concatenate_images_horizontally(images_list, output_type="np"):

    concatenated_images = []

    length = len(images_list[0])
    for i in range(length):
        tmp_tuple = ()
        for item in images_list:
            tmp_tuple += (np.array(item[i]), )

        # Concatenate arrays horizontally
        concatenated_img = np.concatenate(tmp_tuple, axis=1)

        # Convert back to PIL Image
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        elif output_type == "np":
            pass
        else:
            raise NotImplementedError
        concatenated_images.append(concatenated_img)
    return concatenated_images

def read_video_with_mask(video_path, fg_video_path, masks, skip_frames_start=0, skip_frames_end=-1, fps=0):
    '''
    read the video and masks, and return the video, masked video and binary masks
    Args:
        video_path: str, the path of the video
        masks: np.ndarray, the masks of the video
        mask_id: int, the id of the mask
        skip_frames_start: int, the number of frames to skip at the beginning
        skip_frames_end: int, the number of frames to skip at the end
    Returns:
        video: List[Image.Image], the video (RGB)
        masked_video: List[Image.Image], the masked video (RGB)
        binary_masks: List[Image.Image], the binary masks (RGB)
    '''
    to_tensor = ToTensor()
    video = load_video(video_path)[skip_frames_start:skip_frames_end]
    mask = load_video(masks)[skip_frames_start:skip_frames_end]
    fg_video = load_video(fg_video_path)[skip_frames_start:skip_frames_end]     # 100, (576, 576)
    # read fps
    if fps == 0:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
    masked_video = []
    binary_masks = []
    fg_resized = []
    fgy_resized = []
    # 初始化高斯滤波器
    gaussian_filter = quick_freeze(get_gaussian_kernel(
        kernel_size=51,
        sigma=10,
        channels=1
    )).to("cuda")


    for frame, frame_mask, fg_frame in zip(video, mask, fg_video):
        frame_array = np.array(frame)       # (1080, 1920, 3)
        fg_frame_array = np.array(fg_frame)
        target_height, target_width = frame_array.shape[0], frame_array.shape[1]        # 1080, 1920
        
        # 二值化处理
        frame_mask_array = np.array(frame_mask)
        if len(frame_mask_array.shape) == 3:
            frame_mask_array = cv2.cvtColor(frame_mask_array, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(frame_mask_array, 128, 255, cv2.THRESH_BINARY)   # (1080, 1920, 3),   255 | 0

        # 形态学开运算去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        # 连通区域分析，去除小面积区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, 8, cv2.CV_32S)
        filtered_mask = np.zeros_like(cleaned_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > target_height * target_width * 0.0001:
                filtered_mask[labels == i] = 255
        # -------------------------------------------------------------------------------------------------------------------
        # 转换为PyTorch张量并应用高斯滤波
        frame_tensor = to_tensor(filtered_mask / 255.0).unsqueeze(0).to("cuda").float()
        with torch.no_grad():
            filtered_tensor = gaussian_filter(frame_tensor)
        # filtered_tensor = frame_tensor
        # -------------------------------------------------------------------------------------------------------------------
        binary_mask = (filtered_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        black_frame = np.zeros_like(frame_array)

        # binary_mask = (frame_mask == mask_id)
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        masked_frame = np.where(binary_mask_expanded, black_frame, frame_array)
        masked_video.append(Image.fromarray(masked_frame.astype(np.uint8)).convert("RGB"))
        binary_mask_image = np.where(binary_mask, 255, 0).astype(np.uint8)
        binary_masks.append(Image.fromarray(binary_mask_image).convert("RGB"))

        height, width = fg_frame_array.shape[0], fg_frame_array.shape[1]        # 576, 576

        fg_frame_resized = cv2.resize(
                        fg_frame_array, 
                        (target_height, target_height), 
                        interpolation=cv2.INTER_CUBIC  # 也可以选 INTER_LINEAR 或 INTER_NEAREST
                        )
        # -------------------------------------------------------------------------------------------------------------------
        # fg_resizedt = torch.from_numpy(fg_frame_resized).permute(2, 0, 1)
        # fg_resizedt, used_gamma = apply_consistent_gamma(fg_resizedt, gamma_range=(0.7, 0.7))       # gamma启动!
        # fg_frame_resized = fg_resizedt.permute(1, 2, 0).numpy()
        # -------------------------------------------------------------------------------------------------------------------
        pad_width = target_width - fg_frame_resized.shape[1]  # 1920 - 1080 = 840
        pad_left = pad_width // 2           # 420
        # 2. 填充张量（填充值为 255）
        fg_frame_final = np.full(
                (target_height, target_width, fg_frame_resized.shape[2]),
                255,  # 填充值（白色）
                dtype=fg_frame_resized.dtype
            )
        fg_frame_final[:, pad_left : pad_left + fg_frame_resized.shape[1], :] = fg_frame_resized
        fg_resized.append(Image.fromarray(fg_frame_final).convert("RGB"))

        gray = np.dot(fg_frame_final[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        gray_3ch = np.stack((gray, gray, gray), axis=-1)
        fgy_resized.append(Image.fromarray(gray_3ch))


    video = [item.convert("RGB") for item in video]
    return video, masked_video, binary_masks, fps, fg_resized, fgy_resized


accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
    )

def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["i2v_inpainting"],  # i2v_inpainting
    seed: int = 42,
    # inpainting
    inpainting_branch: str = None,
    inpainting_frames: int = None,
    mask_background: bool = False,
    add_first: bool = False,
    first_frame_gt: bool = False,
    replace_gt: bool = False,
    mask_add: bool = False,
    down_sample_fps: int = 8,
    overlap_frames: int = 0,
    prev_clip_weight: float = 0.0,
    start_frame: int = 0,
    end_frame: int = 100,
    img_inpainting_model: str = None,
    llm_model: str = None,
    long_video: bool = False,
    dilate_size: int = -1,
    id_adapter_resample_learnable_path: str = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    # inpainting
    - inpainting_branch (str): The path of the inpainting branch.
    - inpainting_frames (int): The number of frames to generate for inpainting.
    - mask_background (bool): Whether to mask the background.
    - add_first (bool): Whether to add the first frame.
    - first_frame_gt (bool): Whether to use the first frame as the ground truth.
    - replace_gt (bool): Whether to replace the ground truth.
    - mask_add (bool): Whether to add the mask.
    - down_sample_fps (int): The down sample fps.
    """

    video = None
    fps = 16

    print(f"Using the provided inpainting branch: {inpainting_branch}")
    branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to(dtype=dtype).cuda()
    print('Finished load branch')

    transformer = CogVideoXTransformer3D3BModel_sep.from_pretrained(
        "../ckpts/model",
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(dtype=dtype).cuda()

    print('Finished load transformer')
    pipe = CogVideoXI2VTriInpaintPipeline_sep.from_pretrained(
        model_path,
        branch=unwrap_model(branch),
        transformer=unwrap_model(transformer),
        torch_dtype=dtype,
    )
    print('Finished loading !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")


    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    prompt = ""

    video_path = args.video_path          # background video
    mask_path = args.mask_path         # produced mask video
    fg_video_path = args.fg_video_path            # element video
        
    print("-"*100)
    print(f"video_path: {video_path}; start_frame: {start_frame}; end_frame: {end_frame}")
    print("-"*100)
        
    video, masked_video, binary_masks, fps, fg_video, fgy_video = read_video_with_mask(video_path, fg_video_path=fg_video_path, skip_frames_start=start_frame, skip_frames_end=end_frame, masks=mask_path, fps=fps)
    
    
    if generate_type == "i2v_inpainting":
        frames = inpainting_frames
        # down_sample_fps = fps if down_sample_fps == 0 else down_sample_fps
        down_sample_fps = fps#//2
        video, masked_video, binary_masks, fg_video, fgy_video = video[::int(fps//down_sample_fps)], masked_video[::int(fps//down_sample_fps)], binary_masks[::int(fps//down_sample_fps)], fg_video[::int(fps//down_sample_fps)], fgy_video[::int(fps//down_sample_fps)]

        video, masked_video, binary_masks, fg_video, fgy_video = video[:frames], masked_video[:frames], binary_masks[:frames], fg_video[:frames], fgy_video[:frames]
        
        if len(video) < frames:
            raise ValueError(f"video length is less than {frames}, len(video): {len(video)}, using {len(video) - len(video) % 4 + 1} frames...")

        inpaint_outputs = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            video=masked_video,
            masked_video=masked_video,
            mask_video=binary_masks,
            fg_video=fg_video,
            strength=1.0,
            replace_gt=replace_gt,
            mask_add=mask_add,
            id_pool_resample_learnable=False,
            output_type="np"
        ).frames[0]

        torch.cuda.empty_cache()
        
        video_generate = inpaint_outputs

        round_video, final_video = _visualize_video(pipe, mask_background, video[:len(video_generate)], video_generate, binary_masks[:len(video_generate)], fg_video[:len(video_generate)])

        output_dir = './results'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, args.output_path)
        export_to_video(round_video, output_path, fps=8)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    # parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument("--prompt", type=str, default=None, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="../ckpts/CogVideoX-5b-I2V", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--mask_path", type=str, default=None,
        help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--fg_video_path", type=str, default=None,
        help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=10, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="i2v_inpainting", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v', 'inpainting')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--inpainting_branch", type=str, default="../ckpts/branch", help="The path of the inpainting branch")
    parser.add_argument("--inpainting_frames", type=int, default=49, help="The number of frames to generate")
    parser.add_argument(
        "--mask_background",
        action='store_true',
        help="Enable mask_background feature. Default is False.",
    )
    parser.add_argument(
        "--add_first",
        action='store_true',
        help="Enable add_first feature. Default is False.",
    )
    parser.add_argument(
        "--first_frame_gt",
        action='store_true',
        help="Enable first_frame_gt feature. Default is False.",
    )
    parser.add_argument(
        "--replace_gt",
        action='store_true',
        help="Enable replace_gt feature. Default is False.",
    )
    parser.add_argument(
        "--mask_add",
        default=True,
        help="Enable mask_add feature. Default is False.",
    )
    parser.add_argument(
        "--down_sample_fps",
        type=int,
        default=0,
        help="The down sample fps for the video. Default is 8.",
    )
    parser.add_argument(
        "--overlap_frames",
        type=int,
        default=0,
        help="The overlap_frames for the video. Default is 0.",
    )
    parser.add_argument(
        "--prev_clip_weight",
        type=float,
        default=0.0,
        help="The weight for prev_clip. Default is 0.0.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="The start frame.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=100,
        help="The end frame.",
    )
    parser.add_argument(
        "--img_inpainting_model",
        type=str,
        default=None,
        help="The path of the image inpainting model. Default is None.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="The path of the llm model. Default is None.",
    )
    parser.add_argument(
        "--long_video",
        action='store_true',
        help="Enable long_video feature. Default is False.",
    )
    parser.add_argument(
        "--dilate_size",
        type=int,
        default=-1,
        help="The dilate size for the mask. Default is -1.",
    )
    parser.add_argument(
        "--id_adapter_resample_learnable_path",
        type=str,
        default=None,
        help="The path of the id_pool_resample_learnable. Default is None.",
    )
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        inpainting_branch=args.inpainting_branch,
        inpainting_frames=args.inpainting_frames,
        mask_background=args.mask_background,
        add_first=args.add_first,
        first_frame_gt=args.first_frame_gt,
        replace_gt=args.replace_gt,
        mask_add=args.mask_add,
        down_sample_fps=args.down_sample_fps,
        overlap_frames=args.overlap_frames,
        prev_clip_weight=args.prev_clip_weight,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        img_inpainting_model=args.img_inpainting_model,
        llm_model=args.llm_model,
        long_video=args.long_video,
        dilate_size=args.dilate_size,
        id_adapter_resample_learnable_path=args.id_adapter_resample_learnable_path,
    )

# --start_frame    --end_frame