import gradio as gr
import cv2
import os
import tempfile
import numpy as np
from pathlib import Path
import shutil
import json
import torch
from diffusers import (
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3D3BModel_sep,
    CogVideoXI2VTriInpaintPipeline_sep,
)
from accelerate import Accelerator
from diffusers.utils import export_to_video, load_video
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn as nn
from diffusers.utils.torch_utils import is_compiled_module
import random
import logging
import sys 
sys.path.append("..")
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TORCH_CUDA_ARCH_LIST to suppress warning
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.0 7.5 8.0 8.6 8.9 9.0"

accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")

def wrap_model(model):
    model = accelerator.prepare(model)
    return model

def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

# Pre-load models
dtype = torch.bfloat16
model_path = "../ckpts/CogVideoX-5b-I2V"
inpainting_branch = "../ckpts/branch"
transformer_path = "../ckpts/model"

branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to("cuda", dtype=dtype)
transformer = CogVideoXTransformer3D3BModel_sep.from_pretrained(
    transformer_path,
    subfolder="transformer",
    torch_dtype=dtype,
).to("cuda", dtype=dtype)
pipe = CogVideoXI2VTriInpaintPipeline_sep.from_pretrained(
    model_path,
    branch=unwrap_model(branch),
    transformer=unwrap_model(transformer),
    torch_dtype=dtype,
)
pipe.text_encoder.requires_grad_(False)
pipe.transformer.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.branch.requires_grad_(False)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe = wrap_model(pipe)
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Initialize SAM2 models for point-based segmentation
sam2_checkpoint = "../ckpts/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# -------- Helpers --------
def get_first_frame(video_path):
    """获取视频的第一帧"""
    if not video_path or not os.path.exists(video_path):
        return None
        
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame  # Keep in BGR for OpenCV compatibility
    return None

def passthrough(video_path):
    return video_path

def on_gallery_select(evt: gr.SelectData):
    return evt.index

def pick_video_by_index(idx, videos):
    if idx is None:
        return None
    if isinstance(idx, (list, tuple)):
        idx = idx[0] if len(idx) > 0 else None
    if idx is None:
        return None
    if 0 <= idx < len(videos):
        return videos[idx]
    return None

def ensure_directory_exists(path):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_temp_files(paths):
    """清理临时文件"""
    for path in paths:
        try:
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    os.unlink(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"清理临时文件失败 {path}: {e}")

def extract_frames(video_path, output_dir, start_frame=0, end_frame=100):
    """从视频中提取帧"""
    ensure_directory_exists(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break
            
        if frame_count >= start_frame:
            frame_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            
        frame_count += 1
        
    cap.release()
    return frame_count

def create_video_from_frames(frames_dir, output_path, fps=24):
    """从帧创建视频"""
    frame_names = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    if not frame_names:
        return False
        
    first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_name in frame_names:
        frame = cv2.imread(os.path.join(frames_dir, frame_name))
        out.write(frame)
        
    out.release()
    return True

# -------- Video Processing Functions --------
def adjust_video_resolution(video_path, target_width=720, target_height=480, target_frames=100, target_fps=24):
    """调整视频分辨率、帧数和帧率，直接取前100帧"""
    if not video_path or not os.path.exists(video_path):
        return None, "Please provide a valid video file"

    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_output_path = temp_output.name
    temp_output.close()

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Unable to open video file"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, target_fps, (target_width, target_height))

        frame_count = 0
        last_frame = None

        while frame_count < target_frames:
            ret, frame = cap.read()

            if not ret:
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    break

            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)
            frame_count += 1
            last_frame = frame

        cap.release()
        out.release()

        return temp_output_path, "Adjustment successful"
    except Exception as e:
        return None, f"Adjustment Failed: {str(e)}"

def process_foreground_video(fg_video, fg_points):
    """使用SAM2处理前景视频，基于用户点击的点"""
    if not fg_video or not fg_points:
        return None, None, None, "Please provide a foreground video and at least one click point"

    logger.info(f"Begin processing foreground video: {fg_video}")
    logger.info(f"Foreground Point: {fg_points}")

    # 调整视频分辨率
    adjusted_fg_video, status = adjust_video_resolution(
        fg_video,
        target_width=720,
        target_height=480,
        target_frames=100,
        target_fps=24
    )

    if not adjusted_fg_video:
        return None, None, None, f"Foreground video adjustment failed: {status}"

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="fg_processing_")
    frames_dir = os.path.join(temp_dir, "frames")
    mask_dir = os.path.join(temp_dir, "masks")
    element_dir = os.path.join(temp_dir, "elements")
    
    ensure_directory_exists(frames_dir)
    ensure_directory_exists(mask_dir)
    ensure_directory_exists(element_dir)
    
    try:
        # 提取视频帧
        frame_count = extract_frames(adjusted_fg_video, frames_dir, 0, 100)
        if frame_count == 0:
            return None, None, None, "Unable to extract video frame"
            
        # 初始化视频预测器状态
        inference_state = video_predictor.init_state(video_path=frames_dir)
        
        # 处理第一帧
        first_frame_path = os.path.join(frames_dir, "00000.jpg")
        image = Image.open(first_frame_path).convert("RGB")
        image_predictor.set_image(np.array(image))
        
        # 转换点击的点为numpy数组
        point_coords = np.array(fg_points, dtype=np.float32)
        point_labels = np.ones(len(fg_points), dtype=np.int32)  # 正点
        
        # 获取第一帧的掩膜
        masks, scores, logits = image_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        # 注册点到视频预测器
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_coords,
            labels=point_labels,
        )
        
        # 传播以获取所有帧的分割
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # 处理所有帧，创建掩码和前景元素
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        for frame_idx, frame_file in enumerate(frame_files):
            if frame_idx not in video_segments:
                continue
                
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            # 获取当前帧的掩码
            segments = video_segments[frame_idx]
            masks = list(segments.values())
            if not masks:
                continue
                
            mask = np.concatenate(masks, axis=0)[0]  # 假设单个对象
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # 保存掩码
            mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.png")
            cv2.imwrite(mask_path, mask_uint8)
            
            # 创建前景元素（白色背景）
            element = frame.copy()
            element[mask == 0] = 255  # 将背景设为白色
            
            # 保存前景元素
            element_path = os.path.join(element_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(element_path, element)
        
        # 创建前景元素视频
        element_video_path = os.path.join(temp_dir, "element_video.mp4")
        if not create_video_from_frames(element_dir, element_video_path, fps=24):
            return None, None, None, "Failed to create foreground element video"
            
        # 创建掩码视频
        mask_video_path = os.path.join(temp_dir, "mask_video.mp4")
        if not create_video_from_frames(mask_dir, mask_video_path, fps=24):
            return None, None, None, "Failed to create mask video"
            
        return adjusted_fg_video, mask_video_path, element_video_path, "Foreground video processing successful"
            
    except Exception as e:
        error_msg = f"Foreground video processing failed: {str(e)}"
        logger.error(error_msg)
        return None, None, None, error_msg
    finally:
        # 延迟清理 adjusted_fg_video，直到整个流程完成
        pass  # 移除 cleanup_temp_files([adjusted_fg_video])

def auto_adjust_background_video(video_path):
    """自动调整背景视频"""
    if not video_path:
        return None, "Please provide background video"

    adjusted_video, status = adjust_video_resolution(
        video_path,
        target_width=720,
        target_height=480,
        target_frames=100,
        target_fps=24
    )

    return adjusted_video, status

def auto_adjust_foreground_video(video_path):
    """自动调整前景视频"""
    if not video_path:
        return None, "Please provide foreground video"

    adjusted_video, status = adjust_video_resolution(
        video_path,
        target_width=720,
        target_height=480,
        target_frames=100,
        target_fps=24
    )

    return adjusted_video, status

def draw_and_save_trajectory(source_video_path, trajectory_points_json, trajectory_txt_path, target_width=720,
                             target_height=480):
    """绘制并保存轨迹，通过Gradio点击关键点连接成线，使用处理后的分辨率"""
    adjusted_video, _ = adjust_video_resolution(source_video_path, target_width, target_height)
    if not adjusted_video:
        return None, None, "Unable to adjust video resolution"

    cap = cv2.VideoCapture(adjusted_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None, None, "Unable to read video file"

    first_frame = cv2.resize(first_frame, (target_width, target_height))

    trajectory_points = []
    if trajectory_points_json:
        try:
            trajectory_points = json.loads(trajectory_points_json)
        except:
            cap.release()
            return None, None, "无效的轨迹点数据"

    if not trajectory_points:
        cap.release()
        return None, None, "未选择任何关键点"

    if len(trajectory_points) == 1:
        prev_frame = first_frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_point = np.array([[trajectory_points[0]]], dtype=np.float32)
        flow_points = [tuple(trajectory_points[0])]

        for _ in range(total_frames - 1):
            ret, next_frame = cap.read()
            if not ret:
                break
            next_frame = cv2.resize(next_frame, (target_width, target_height))
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_point, None
            )
            if status[0] == 1:
                new_point = tuple(map(int, next_point[0][0]))
                flow_points.append(new_point)
                prev_point = next_point
                prev_gray = next_gray
            else:
                flow_points.append(flow_points[-1])

        trajectory_points = flow_points[:total_frames]

    elif len(trajectory_points) >= 2:
        interpolated_points = []
        segments = len(trajectory_points) - 1
        points_per_segment = total_frames // segments
        for i in range(segments):
            start_point = trajectory_points[i]
            end_point = trajectory_points[i + 1]
            n_points = points_per_segment if i < segments - 1 else (total_frames - (segments - 1) * points_per_segment)
            for j in range(n_points):
                ratio = j / max(1, n_points - 1)
                x = int(start_point[0] + ratio * (end_point[0] - start_point[0]))
                y = int(start_point[1] + ratio * (end_point[1] - start_point[1]))
                interpolated_points.append((x, y))
        trajectory_points = interpolated_points[:total_frames]

    with open(trajectory_txt_path, 'w') as f:
        for point in trajectory_points:
            f.write(f"{point[0]},{point[1]}\n")

    vis_frame = first_frame.copy()
    for i in range(1, len(trajectory_points)):
        cv2.line(vis_frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)
    for point in trajectory_points:
        cv2.circle(vis_frame, point, 5, (0, 255, 0), -1)

    vis_path = trajectory_txt_path.replace('.txt', '_vis.jpg')
    cv2.imwrite(vis_path, vis_frame)

    cap.release()
    if os.path.exists(adjusted_video):
        os.unlink(adjusted_video)
    return vis_path, trajectory_txt_path, f"轨迹已保存至 {trajectory_txt_path}"

def load_trajectory(trajectory_path):
    """加载轨迹坐标文件"""
    trajectory = []
    with open(trajectory_path, 'r') as f:
        for line in f:
            x, y = map(int, line.strip().split(','))
            trajectory.append((x, y))
    return trajectory

def generate_mask_video_with_trajectory(fg_element_path, source_video_path, output_path, trajectory_path, scales=[1.0],
                                        target_width=720, target_height=480, alignment="center"):
    """生成带轨迹的掩膜视频 - 使用前景元素视频而不是掩码视频，支持alignment参数和动态scales"""
    trajectory = load_trajectory(trajectory_path)
    if not trajectory:
        return None, "Unable to load trajectory file"

    fg_cap = cv2.VideoCapture(fg_element_path)
    source_cap = cv2.VideoCapture(source_video_path)

    source_width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = source_cap.get(cv2.CAP_PROP_FPS)

    frame_count = min(
        int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(trajectory)
    )

    # 过滤有效的 scales（非 None 且非 0），并沿用最后一个有效值填充后续未指定的 scales
    valid_scales = [s for s in scales if s is not None and s != 0]
    if not valid_scales:
        valid_scales = [1.0]  # 默认值
    else:
        last_valid = valid_scales[-1]
        valid_scales.extend([last_valid] * (len(scales) - len(valid_scales)))  # 填充未指定的 scales

    n_scales = len(valid_scales)
    if n_scales == 0:
        valid_scales = [1.0]
        n_scales = 1

    # 计算关键帧位置，仅考虑有效 scales
    if n_scales > 1:
        key_frames = [int(i * (frame_count - 1) / (n_scales - 1)) for i in range(n_scales)]
    else:
        key_frames = [0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mask = cv2.VideoWriter(output_path, fourcc, fps, (source_width, source_height), isColor=False)

    frame_num = 0
    while frame_num < frame_count:
        ret_fg, fg_frame = fg_cap.read()
        ret_source, source_frame = source_cap.read()

        if not ret_fg or not ret_source:
            break

        center_x, center_y = trajectory[frame_num]

        # 计算当前 scale
        if n_scales == 1:
            current_scale = valid_scales[0]
        else:
            for j in range(1, n_scales):
                if frame_num <= key_frames[j]:
                    left = key_frames[j-1]
                    right = key_frames[j]
                    left_scale = valid_scales[j-1]
                    right_scale = valid_scales[j]
                    if left == right:
                        current_scale = left_scale
                    else:
                        ratio = (frame_num - left) / (right - left)
                        current_scale = left_scale + ratio * (right_scale - left_scale)
                    break
            else:
                current_scale = valid_scales[-1]

        gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros((source_height, source_width), dtype=np.uint8)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            if current_scale != 1.0:
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    scaling_matrix = np.array([[current_scale, 0, (1 - current_scale) * cx],
                                               [0, current_scale, (1 - current_scale) * cy]])
                    scaled_contour = cv2.transform(largest_contour, scaling_matrix).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(scaled_contour)
                else:
                    scaled_contour = largest_contour.astype(np.int32)
            else:
                scaled_contour = largest_contour.astype(np.int32)

            if alignment == "center":
                start_x = center_x - w // 2
                start_y = center_y - h // 2
            elif alignment == "bottom":
                start_x = center_x - w // 2
                start_y = center_y - h
            else:
                raise ValueError(f"Unknown alignment: {alignment}")

            target_x1 = max(0, start_x)
            target_y1 = max(0, start_y)
            target_x2 = min(source_width, start_x + w)
            target_y2 = min(source_height, start_y + h)

            src_x1 = max(0, -start_x)
            src_y1 = max(0, -start_y)
            src_x2 = w - max(0, (start_x + w) - source_width)
            src_y2 = h - max(0, (start_y + h) - source_height)

            contour_offset = np.array([target_x1 - src_x1, target_y1 - src_y1])

            current_contour = scaled_contour if current_scale != 1.0 else largest_contour.astype(np.int32)
            shifted_contour = current_contour + contour_offset - np.array([x, y])

            cv2.fillPoly(final_mask, [shifted_contour.astype(np.int32)], 255)

        out_mask.write(final_mask)

        frame_num += 1

    fg_cap.release()
    source_cap.release()
    out_mask.release()
    return output_path, "Mask video generated successfully"

def quick_freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def apply_consistent_gamma(frames, gamma=None, gamma_range=(0.5, 2)):
    frames_normalized = frames.float() / 255.0
    if gamma is None:
        gamma = random.uniform(*gamma_range)
    corrected_frames = torch.pow(frames_normalized, gamma) * 255.0
    return corrected_frames.to(frames.dtype), gamma

def get_gaussian_kernel(kernel_size, sigma, channels):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def read_video_with_mask(video_path, fg_video_path, masks, skip_frames_start=0, skip_frames_end=-1, fps=0):
    to_tensor = ToTensor()
    video = load_video(video_path)[skip_frames_start:skip_frames_end]
    mask = load_video(masks)[skip_frames_start:skip_frames_end]
    fg_video = load_video(fg_video_path)[skip_frames_start:skip_frames_end]
    if fps == 0:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    masked_video = []
    binary_masks = []
    fg_resized = []
    fgy_resized = []
    gaussian_filter = quick_freeze(get_gaussian_kernel(kernel_size=51, sigma=10, channels=1)).to("cuda")
    for frame, frame_mask, fg_frame in zip(video, mask, fg_video):
        frame_array = np.array(frame)
        fg_frame_array = np.array(fg_frame)
        target_height, target_width = frame_array.shape[0], frame_array.shape[1]
        frame_mask_array = np.array(frame_mask)
        if len(frame_mask_array.shape) == 3:
            frame_mask_array = cv2.cvtColor(frame_mask_array, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(frame_mask_array, 128, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, 8, cv2.CV_32S)
        filtered_mask = np.zeros_like(cleaned_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > target_height * target_width * 0.0001:
                filtered_mask[labels == i] = 255
        frame_tensor = to_tensor(filtered_mask / 255.0).unsqueeze(0).to("cuda").float()
        with torch.no_grad():
            filtered_tensor = gaussian_filter(frame_tensor)
            # filtered_tensor = frame_tensor
        binary_mask = (filtered_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        black_frame = np.zeros_like(frame_array)
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        masked_frame = np.where(binary_mask_expanded, black_frame, frame_array)
        masked_video.append(Image.fromarray(masked_frame.astype(np.uint8)).convert("RGB"))
        binary_mask_image = np.where(binary_mask, 255, 0).astype(np.uint8)
        binary_masks.append(Image.fromarray(binary_mask_image).convert("RGB"))
        height, width = fg_frame_array.shape[0], fg_frame_array.shape[1]
        fg_frame_resized = cv2.resize(fg_frame_array, (target_height, target_height), interpolation=cv2.INTER_CUBIC)
        pad_width = target_width - fg_frame_resized.shape[1]
        pad_left = pad_width // 2
        fg_frame_final = np.full((target_height, target_width, fg_frame_resized.shape[2]), 255,
                                dtype=fg_frame_resized.dtype)
        fg_frame_final[:, pad_left:pad_left + fg_frame_resized.shape[1], :] = fg_frame_resized
        fg_resized.append(Image.fromarray(fg_frame_final).convert("RGB"))
        gray = np.dot(fg_frame_final[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        gray_3ch = np.stack((gray, gray, gray), axis=-1)
        fgy_resized.append(Image.fromarray(gray_3ch))
    video = [item.convert("RGB") for item in video]
    return video, masked_video, binary_masks, fps, fg_resized, fgy_resized

def generate_video(
        output_path: str = "./output.mp4",
        video_path: str = "",
        mask_path: str = "",
        fg_video_path: str = "",
        num_inference_steps: int = 10,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        generate_type: str = "i2v_inpainting",
        seed: int = 42,
        inpainting_frames: int = 49,
        mask_background: bool = False,
        add_first: bool = False,
        first_frame_gt: bool = False,
        replace_gt: bool = False,
        mask_add: bool = True,
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
    fps = 24
    # 使用固定提示词
    prompt = "A realistic scene with objects moving naturally"
    
    try:
        # Validate input paths
        if not os.path.exists(video_path):
            raise ValueError(f"Video path {video_path} does not exist")
        if not os.path.exists(mask_path):
            raise ValueError(f"Mask path {mask_path} does not exist")
        if not os.path.exists(fg_video_path):
            raise ValueError(f"Foreground video path {fg_video_path} does not exist")

        video, masked_video, binary_masks, fps, fg_video, fgy_video = read_video_with_mask(
            video_path, fg_video_path=fg_video_path, skip_frames_start=start_frame, skip_frames_end=end_frame,
            masks=mask_path, fps=fps
        )

        if generate_type == "i2v_inpainting":
            frames = inpainting_frames
            down_sample_fps = fps // 2
            video, masked_video, binary_masks, fg_video, fgy_video = (
                video[::int(fps // down_sample_fps)],
                masked_video[::int(fps // down_sample_fps)],
                binary_masks[::int(fps // down_sample_fps)],
                fg_video[::int(fps // down_sample_fps)],
                fgy_video[::int(fps // down_sample_fps)],
            )
            video, masked_video, binary_masks, fg_video, fgy_video = (
                video[:frames],
                masked_video[:frames],
                binary_masks[:frames],
                fg_video[:frames],
                fgy_video[:frames],
            )
            if len(video) < frames:
                raise ValueError(f"video length is less than {frames}, len(video): {len(video)}")
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
            output_dir = './results'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            export_to_video(video_generate, output_path, fps=12)
            return output_path, "Video compositing successful"
        else:
            raise NotImplementedError
    except Exception as e:
        return None, f"Video compositing failed: {str(e)}"
    finally:
        torch.cuda.empty_cache()

# -------- Your samples --------
sample_videos = [
    "../assets/bg/511307.mp4",
    "../assets/bg/100115.mp4",
    "../assets/fg/44867.mp4",
    "../assets/fg/45871.mp4",
    "../assets/fg/47853.mp4",
    "../assets/bg/42858.mp4",
]

thumbnails = [get_first_frame(p) for p in sample_videos]
cards = []
for i, (thumb, path) in enumerate(zip(thumbnails, sample_videos)):
    title = os.path.basename(path)
    if thumb is not None:
        thumb_rgb = thumb[:, :, ::-1]
        cards.append([thumb_rgb, f"{i + 1}. {title}"])

# -------- Gradio Interface --------
css = """
.composed-video {
    width: 100% !important;
    max-width: 100% !important;
    margin: 20px auto;
    display: block;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.step-group {
    border: 2px solid #4A90E2;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    background-color: #F9FAFB;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    overflow: hidden !important;
}

.rounded-button {
    border-radius: 12px !important;
    width: 180px !important;
    padding: 10px;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.2s ease;
}

.rounded-button:hover {
    transform: translateY(-2px);
    background-color: #add8e6 !important;
}

.bg-button {
    border-radius: 12px !important;
    width: 600px !important;
    margin: 0 auto !important;
    display: block !important;
    padding: 12px;
    font-weight: 600;
}

.custom-button {
    background-color: #add8e6 !important;
    color: black !important;
    border: 1px solid #add8e6 !important;
    font-size: 24px;
}

.custom-button:hover {
    background-color: #add8e6 !important;
}

.step5-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto;
    padding: 20px;
}

.step5-video {
    width: 100% !important;
    max-width: 100% !important;
    border-radius: 8px;
}

.step5-title {
    text-align: center !important;
    font-size: 2.5em;
    color: #1F2937;
    margin-bottom: 20px;
    font-weight: 700;
}

.video-display {
    height: 360px !important;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    background-color: white !important;
}

.image-display {
    height: 360px !important;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    background-color: white !important;
}

.video-container, .image-container {
    background-color: white !important;
    padding: 10px;
    border-radius: 8px;
}

gradio-container video, gradio-container img {
    background-color: white !important;
    border-radius: 6px;
}

gradio-container {
    background-color: #F3F4F6 !important;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    color: #1F2937;
    text-align: center;
    letter-spacing: 1px;
    margin-bottom: 30px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}

h1 {
    font-size: 2.5em;
    font-weight: 800;
    background: linear-gradient(45deg, #2563EB, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2 {
    font-size: 1.8em;
    font-weight: 700;
    border-bottom: 2px solid #4A90E2;
    padding-bottom: 10px;
}

.step-number {
    color: #FF4500; /* Deep orange color for Step numbers */
}

.bold-text {
    font-weight: bold;
}

/* 新增样式 */
.button-spacing {
    gap: 20px !important;
}

.gallery-column {
    flex: 4 !important;
    overflow: auto;
}

.preview-column {
    flex: 3 !important;
}

.quick-guide {
    font-size: 1.2em;
    font-weight: bold;
    text-align: left;
    margin-bottom: 10px;
}

.label-text {
    font-size: 1.2em;
    font-weight: 600;
    color: #1F2937;
    margin-bottom: 2px;
    margin-top: 5px;
    text-align: left;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # GenCompositor Demo
        ### Create stunning video compositing with ease.

        <span class='quick-guide'>💡 Quick Guide</span>
        - We provide some video samples for you. You can choose to use them as foreground or background footage (set '**Yes**'), or upload your own videos (set '**No**').
        - **<span class='step-number'>Step 1</span>**: Click the "**Begin selecting object**" button, then click on the first frame of the foreground video to select the desired element. Click "**Segment foreground element**" to segment it.
        - **<span class='step-number'>Step 2</span>**: Click the "**Begin specifying trajectory**" button, then click on the first frame of the background video to design the desired trajectory. Click "**Complete trajectory drawing**" to finish.
        - **<span class='step-number'>Step 3</span>**: Set the "**Rescale parameters**" and "**Inference steps number**" (default: 10), then click "**Generate middle mask for compositing**" to create the mask video. Click "**Generate composited video**" to obtain the results.
        - **<span class='step-number'>(Optional)</span>**: To remove an object from the video, set "**Inference steps number**" (default: 10), click "**Generate mask for removal**" and then "**Generate removed video**" in the "Extended Application" section.
        """
    )

    processed_bg_video = gr.State(None)
    processed_fg_element = gr.State(None)
    processed_fg_mask = gr.State(None)
    processed_fg_source = gr.State(None)
    trajectory_file = gr.State(None)
    selected_sample_video = gr.State(None)
    trajectory_visualization = gr.State(None)
    trajectory_points = gr.State([])
    original_bg_frame = gr.State(None)
    final_mask_output_state = gr.State(None)
    is_drawing = gr.State(False)
    fg_points = gr.State([])
    is_drawing_fg = gr.State(False)
    original_fg_frame = gr.State(None)
    inference_steps = gr.State(10)  # New state for inference steps

    with gr.Group(elem_classes="step-group"):
        gr.Markdown("### Using our video samples?")
        use_samples = gr.Radio(choices=["Yes", "No"], value="Yes", label="", interactive=True)

        with gr.Row(visible=True) as samples_row:
            with gr.Column(scale=4, elem_classes="gallery-column"):
                gallery = gr.Gallery(
                    value=cards,
                    label="Video Samples. Click to select the sample video and it will play on the right.",
                    show_label=True,
                    columns=[3],
                    height="360px",
                )
            with gr.Column(scale=3, elem_classes="preview-column"):
                sample_preview = gr.Video(
                    label="Selected Video Preview. Click buttons below to determine whether it is foreground or background",
                    interactive=False,
                    height=360,
                    elem_classes="video-display"
                )
                with gr.Row():
                    select_fg_gallery = gr.Button(
                        "Select as foreground",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"],
                        visible=True
                    )
                    select_bg_gallery = gr.Button(
                        "Select as background",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"],
                        visible=True
                    )

    selected_idx = gr.State(None)

    def update_sample_preview(evt: gr.SelectData):
        idx = evt.index
        if idx is None:
            return None, None, None

        if 0 <= idx < len(sample_videos):
            video_path = sample_videos[idx]
            return video_path, video_path, idx
        return None, None, None

    gallery.select(
        fn=update_sample_preview,
        inputs=None,
        outputs=[selected_sample_video, sample_preview, selected_idx]
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("## <span class='step-number'>Step 1</span>: Foreground Video Processing")
                with gr.Row(visible=False) as fg_upload_row:
                    upload_fg_button = gr.UploadButton(
                        "Upload your foreground video",
                        file_types=[".mp4"],
                        variant="primary",
                        elem_classes=["custom-button", "bg-button"]
                    )
                fg_video_input = gr.Video(label="Foreground video")
                with gr.Row(elem_classes="button-spacing"):
                    btn_start_drawing_fg = gr.Button(
                        "Begin selecting object",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                    btn_clear_fg_points = gr.Button(
                        "Clear selected points",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                fg_status = gr.Textbox(label="Foreground video processing status", interactive=False)
                fg_first_frame = gr.Image(
                    label="Please click to select an object",
                    interactive=True,
                    width=900,
                    height=360,
                    elem_classes="image-display"
                )
                with gr.Row():
                    btn_process_fg = gr.Button(
                        "Segment foreground element",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                fg_element_output = gr.Video(
                    label="Foreground element video",
                    height=360,
                    elem_classes="video-display",
                    format="mp4"
                )

                def update_fg_video(video_path):
                    if not video_path:
                        return None, None, None, None, "Please provide foreground video", [], None
                    adjusted_video, status = auto_adjust_foreground_video(video_path)
                    frame = get_first_frame(adjusted_video)
                    frame_rgb = cv2.resize(frame, (720, 480))[:, :, ::-1] if frame is not None else None
                    return None, None, adjusted_video, frame_rgb, status, [], frame_rgb

                fg_video_input.change(
                    fn=update_fg_video,
                    inputs=[fg_video_input],
                    outputs=[fg_element_output, processed_fg_element, processed_fg_source, fg_first_frame, fg_status, fg_points, original_fg_frame]
                )

                select_fg_gallery.click(
                    fn=lambda idx: pick_video_by_index(idx, sample_videos),
                    inputs=selected_idx,
                    outputs=fg_video_input,
                )

                def process_fg_upload(uploaded_file):
                    return uploaded_file.name if uploaded_file else None

                upload_fg_button.upload(
                    fn=process_fg_upload,
                    inputs=[upload_fg_button],
                    outputs=[fg_video_input]
                )

                def start_drawing_fg():
                    return True, "Begin selecting object"

                btn_start_drawing_fg.click(
                    fn=start_drawing_fg,
                    inputs=[],
                    outputs=[is_drawing_fg, fg_status]
                )
                
                def add_fg_point(evt: gr.SelectData, original_fg_frame, current_points, is_drawing_fg):
                    if not is_drawing_fg:
                        return gr.update(), current_points, "Please click the 'Begin selecting objects' button first"
                    if original_fg_frame is None:
                        return gr.update(), current_points, "Please select foreground element first"

                    x, y = evt.index
                    x = min(max(0, int(x)), 720)
                    y = min(max(0, int(y)), 480)
                    new_points = current_points + [(x, y)]

                    display_frame = original_fg_frame.copy()
                    for point in new_points:
                        cv2.circle(display_frame, point, 8, (0, 255, 0), -1)

                    return display_frame, new_points, f"Added the point ({x}, {y}), a total of {len(new_points)} points"

                fg_first_frame.select(
                    fn=add_fg_point,
                    inputs=[original_fg_frame, fg_points, is_drawing_fg],
                    outputs=[fg_first_frame, fg_points, fg_status]
                )

                def clear_fg_points(original_fg_frame):
                    if not original_fg_frame is None:
                        return original_fg_frame, [], False, "Selected point cleared"
                    return None, [], False, "Please select foreground element first"

                btn_clear_fg_points.click(
                    fn=clear_fg_points,
                    inputs=[original_fg_frame],
                    outputs=[fg_first_frame, fg_points, is_drawing_fg, fg_status]
                )

                def process_fg_and_update_state(fg_video, fg_points):
                    source, mask, element, status = process_foreground_video(fg_video, fg_points)
                    debug_info = f"Path of element video: {element}, existing in: {os.path.exists(element) if element else False}"
                    status_with_debug = f"{status}\n{debug_info}"
                    return element, element, source, mask, status_with_debug

                btn_process_fg.click(
                    fn=process_fg_and_update_state,
                    inputs=[fg_video_input, fg_points],
                    outputs=[fg_element_output, processed_fg_element, processed_fg_source, processed_fg_mask, fg_status]
                )

        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("## <span class='step-number'>Step 2</span>: Specify Trajectory in Background")
                with gr.Row(visible=False) as bg_upload_row:
                    upload_bg_button = gr.UploadButton(
                        "Upload your background video",
                        file_types=[".mp4"],
                        variant="primary",
                        elem_classes=["custom-button", "bg-button"]
                    )
                bg_video_input = gr.Video(label="Background video")
                with gr.Row(elem_classes="button-spacing"):
                    btn_start_drawing = gr.Button(
                        "Begin specifying trajectory",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                    btn_clear_points = gr.Button(
                        "Clear selected points",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                bg_status = gr.Textbox(label="Trajectory designing status", interactive=False)
                bg_first_frame = gr.Image(
                    label="Please click to select trajectory keypoints",
                    interactive=True,
                    width=900,
                    height=360,
                    elem_classes="image-display"
                )
                with gr.Row():
                    btn_draw_trajectory = gr.Button(
                        "Complete trajectory drawing",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                trajectory_vis_output = gr.Image(
                    label="Trajectory visualization",
                    interactive=False,
                    width=900,
                    height=360,
                    elem_classes="image-display"
                )

                def update_bg_video_and_frame(bg_video, original_bg_frame):
                    if not bg_video:
                        return None, None, "Please provide background video", [], None
                    adjusted_video, status = auto_adjust_background_video(bg_video)
                    frame = get_first_frame(adjusted_video)
                    frame_rgb = cv2.resize(frame, (720, 480))[:, :, ::-1] if frame is not None else None
                    return adjusted_video, frame_rgb, status, [], frame_rgb

                bg_video_input.change(
                    fn=update_bg_video_and_frame,
                    inputs=[bg_video_input, original_bg_frame],
                    outputs=[processed_bg_video, bg_first_frame, bg_status, trajectory_points, original_bg_frame]
                )

                select_bg_gallery.click(
                    fn=lambda idx: pick_video_by_index(idx, sample_videos),
                    inputs=selected_idx,
                    outputs=bg_video_input,
                )

                def process_bg_upload(uploaded_file):
                    return uploaded_file.name if uploaded_file else None

                upload_bg_button.upload(
                    fn=process_bg_upload,
                    inputs=[upload_bg_button],
                    outputs=[bg_video_input]
                )

                def start_drawing():
                    return True, "Begin specifying trajectory"

                btn_start_drawing.click(
                    fn=start_drawing,
                    inputs=[],
                    outputs=[is_drawing, bg_status]
                )

                def add_trajectory_point(evt: gr.SelectData, original_bg_frame, current_points, is_drawing):
                    if not is_drawing:
                        return gr.update(), current_points, "Please click the 'Begin specifying trajectory' button first"
                    if original_bg_frame is None:
                        return gr.update(), current_points, "Please select a background video first"

                    x, y = evt.index
                    x = min(max(0, int(x)), 720)
                    y = min(max(0, int(y)), 480)
                    new_points = current_points + [(x, y)]

                    display_frame = original_bg_frame.copy()

                    if len(new_points) > 1:
                        for i in range(1, len(new_points)):
                            cv2.line(display_frame, new_points[i - 1], new_points[i], (0, 0, 255), 3)

                    for point in new_points:
                        cv2.circle(display_frame, point, 8, (0, 255, 0), -1)

                    return display_frame, new_points, f"Added the point ({x}, {y}), a total of {len(new_points)} points"

                bg_first_frame.select(
                    fn=add_trajectory_point,
                    inputs=[original_bg_frame, trajectory_points, is_drawing],
                    outputs=[bg_first_frame, trajectory_points, bg_status]
                )

                def clear_points(original_bg_frame):
                    if not original_bg_frame is None:
                        return original_bg_frame, [], False, "Selected point cleared"
                    return None, [], False, "Please select background video first"

                btn_clear_points.click(
                    fn=clear_points,
                    inputs=[original_bg_frame],
                    outputs=[bg_first_frame, trajectory_points, is_drawing, bg_status]
                )

                def process_trajectory(bg_video, points):
                    if not bg_video:
                        return None, None, None, "Please provide background video"
                    if not points:
                        return None, None, None, "Please select at least one key point"
                    temp_trajectory = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
                    points_json = json.dumps(points)
                    vis_path, trajectory_path, status = draw_and_save_trajectory(bg_video, points_json, temp_trajectory)
                    return vis_path, trajectory_path, vis_path, status

                btn_draw_trajectory.click(
                    fn=process_trajectory,
                    inputs=[processed_bg_video, trajectory_points],
                    outputs=[trajectory_vis_output, trajectory_file, trajectory_vis_output, bg_status]
                )

    def update_interface(use):
        if use == "Yes":
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )

    use_samples.change(
        fn=update_interface,
        inputs=[use_samples],
        outputs=[samples_row, select_fg_gallery, upload_fg_button, select_bg_gallery, upload_bg_button, fg_upload_row, bg_upload_row]
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("## <span class='step-number'>Step 3</span>: Video Compositing")
                compose_status = gr.Textbox(label="Video compositing status", interactive=False)
                gr.Markdown(
                    "If the trajectory is the midpoint of desired mask, click the left; if the trajectory is the low point of desired mask, click the right",
                    elem_classes="label-text"
                )
                with gr.Row(elem_classes="button-spacing"):
                    btn_generate_mask = gr.Button(
                        "Generate middle mask for compositing",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                    btn_generate_upper_mask = gr.Button(
                        "Generate upper mask for compositing",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    rescale1 = gr.Number(
                        label="Rescale 1.",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale2 = gr.Number(
                        label="Rescale 2 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale3 = gr.Number(
                        label="Rescale 3 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale4 = gr.Number(
                        label="Rescale 4 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale5 = gr.Number(
                        label="Rescale 5 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                with gr.Row():
                    final_mask_output = gr.Video(
                        label="Mask for compositing",
                        height=300,
                        elem_classes=["step5-video"],
                        format="mp4"
                    )
                with gr.Row():
                    inference_steps_input = gr.Number(
                        label="Inference steps number. Bigger value brings better performance but costs more time.",
                        value=10,
                        minimum=1,
                        maximum=100,
                        step=1,
                        scale=1
                    )
                with gr.Row(elem_classes="button-spacing"):
                    btn_compose_video = gr.Button(
                        "Generate composited video",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    composed_video_output = gr.Video(
                        label="Composited video",
                        height=300,
                        elem_classes=["step5-video"],
                        autoplay=True,
                        format="mp4"
                    )

                def generate_final_mask(fg_element, bg_video, trajectory_path, rescale_list, alignment="center"):
                    scales = [r for r in rescale_list if r is not None and r != 0]
                    if not scales:
                        scales = [0.4]
                    if not fg_element or not bg_video or not trajectory_path:
                        return None, None, "Please provide complete foreground element video, background video, and specified trajectory"
                    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    mask_video, status = generate_mask_video_with_trajectory(fg_element, bg_video, temp_output,
                                                                           trajectory_path, scales, alignment=alignment)
                    return mask_video, mask_video, status

                btn_generate_mask.click(
                    fn=lambda fg, bg, traj, r1, r2, r3, r4, r5: generate_final_mask(fg, bg, traj, [r1, r2, r3, r4, r5], "center"),
                    inputs=[processed_fg_element, processed_bg_video, trajectory_file, rescale1, rescale2, rescale3, rescale4, rescale5],
                    outputs=[final_mask_output, final_mask_output_state, compose_status]
                )

                btn_generate_upper_mask.click(
                    fn=lambda fg, bg, traj, r1, r2, r3, r4, r5: generate_final_mask(fg, bg, traj, [r1, r2, r3, r4, r5], "bottom"),
                    inputs=[processed_fg_element, processed_bg_video, trajectory_file, rescale1, rescale2, rescale3, rescale4, rescale5],
                    outputs=[final_mask_output, final_mask_output_state, compose_status]
                )

                def compose_video(fg_element, bg_video, mask_video, inference_steps):
                    if not fg_element or not bg_video or not mask_video:
                        return None, "Please ensure that foreground element video, background video and final mask video are provided"
                    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    try:
                        output_path, status = generate_video(
                            output_path=temp_output,
                            video_path=bg_video,
                            mask_path=mask_video,
                            fg_video_path=fg_element,
                            num_inference_steps=int(inference_steps),
                            guidance_scale=6.0,
                            num_videos_per_prompt=1,
                            generate_type="i2v_inpainting",
                            seed=42,
                            inpainting_frames=49,
                            mask_background=False,
                            add_first=False,
                            first_frame_gt=False,
                            replace_gt=False,
                            mask_add=True,
                            down_sample_fps=8,
                            overlap_frames=0,
                            prev_clip_weight=0.0,
                            start_frame=0,
                            end_frame=100,
                            img_inpainting_model=None,
                            llm_model=None,
                            long_video=False,
                            dilate_size=-1,
                            id_adapter_resample_learnable_path=None,
                        )
                        return output_path, status
                    except Exception as e:
                        return None, f"Video compositing failed: {str(e)}"
                    finally:
                        torch.cuda.empty_cache()

                btn_compose_video.click(
                    fn=compose_video,
                    inputs=[processed_fg_element, processed_bg_video, final_mask_output_state, inference_steps_input],
                    outputs=[composed_video_output, compose_status]
                )

        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("### <span class='step-number'>(Optional)</span> Extended Application: Video Inpainting and Removal")
                inpaint_status = gr.Textbox(label="Video element removal status", interactive=False)
                with gr.Row(elem_classes="button-spacing"):
                    btn_generate_mask_removal = gr.Button(
                        "Generate mask for removal",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    removal_mask_output = gr.Video(
                        label="Mask for removal",
                        height=300,
                        elem_classes=["step5-video"],
                        format="mp4"
                    )
                with gr.Row():
                    inference_steps_removal_input = gr.Number(
                        label="Inference steps number. Bigger value brings better performance but costs more time.",
                        value=10,
                        minimum=1,
                        maximum=100,
                        step=1,
                        scale=1
                    )
                with gr.Row(elem_classes="button-spacing"):
                    btn_generate_inpainted = gr.Button(
                        "Generate removed video",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    inpainted_video_output = gr.Video(
                        label="Removed video",
                        height=300,
                        elem_classes=["step5-video"],
                        autoplay=True,
                        format="mp4"
                    )

                def generate_mask_for_removal(processed_fg_mask):
                    if not processed_fg_mask:
                        return None, None, "Please segment foreground element first to generate a mask"
                    return processed_fg_mask, processed_fg_mask, "Mask for removal generated successfully"

                btn_generate_mask_removal.click(
                    fn=generate_mask_for_removal,
                    inputs=[processed_fg_mask],
                    outputs=[removal_mask_output, final_mask_output_state, inpaint_status]
                )

                def compose_inpainted_video(video, mask_video, inference_steps):
                    fg_element = "../assets/fg/white_video.mp4"
                    if not video or not mask_video:
                        return None, "Please ensure that video and mask are provided"
                    if not os.path.exists(video):
                        return None, f"Video path {video} does not exist"
                    if not os.path.exists(mask_video):
                        return None, f"Mask video path {mask_video} does not exist"
                    if not os.path.exists(fg_element):
                        return None, f"Foreground element path {fg_element} does not exist"
                    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    try:
                        output_path, status = generate_video(
                            output_path=temp_output,
                            video_path=video,
                            mask_path=mask_video,
                            fg_video_path=fg_element,
                            num_inference_steps=int(inference_steps),
                            guidance_scale=6.0,
                            num_videos_per_prompt=1,
                            generate_type="i2v_inpainting",
                            seed=42,
                            inpainting_frames=49,
                            mask_background=False,
                            add_first=False,
                            first_frame_gt=False,
                            replace_gt=False,
                            mask_add=True,
                            down_sample_fps=8,
                            overlap_frames=0,
                            prev_clip_weight=0.0,
                            start_frame=0,
                            end_frame=100,
                            img_inpainting_model=None,
                            llm_model=None,
                            long_video=False,
                            dilate_size=-1,
                            id_adapter_resample_learnable_path=None,
                        )
                        return output_path, status
                    except Exception as e:
                        return None, f"Video inpainting failed: {str(e)}"
                    finally:
                        torch.cuda.empty_cache()

                btn_generate_inpainted.click(
                    fn=compose_inpainted_video,
                    inputs=[processed_fg_source, final_mask_output_state, inference_steps_removal_input],
                    outputs=[inpainted_video_output, inpaint_status]
                )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    allowed_paths=["/group/40075/shuzhouyang/GenCompositor-main/assets"]
)