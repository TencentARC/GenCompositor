import os
import cv2
import torch
import numpy as np
import supervision as sv
import time
import argparse
import csv  # 新增：用于读取CSV文件
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms
import torch.nn.functional as F
from easydict import EasyDict as edict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import sys 
sys.path.append("..")
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.video_utils import calculate_max_mask_ratio, process_frame

from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置参数
parser = argparse.ArgumentParser()
# 添加视频和分割参数
parser.add_argument('--root', type=str, default='../assets', help='path to the video')
parser.add_argument('--vid_name', type=str, default='breakdance', help='path to the video')
# 设置GPU
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
# 设置模型
parser.add_argument('--model', type=str, default='cotracker', help='model name')
# 设置下采样因子
parser.add_argument('--downsample', type=float, default=1, help='downsample factor')
parser.add_argument('--grid_size', type=int, default=10, help='grid size')
# 设置输出目录
parser.add_argument('--save_path', type=str, default='../assets/fg/source', help='output directory')
parser.add_argument('--mask_path', type=str, default='../assets/fg/mask', help='output directory')
parser.add_argument('--fg_path', type=str, default='../assets/fg/element', help='output directory')
# 设置帧率
parser.add_argument('--fps', type=float, default=1, help='fps')
# 设置轨迹长度
parser.add_argument('--len_track', type=int, default=0, help='len_track')
parser.add_argument('--fps_vis', type=int, default=12, help='len_track')
# 裁剪视频
parser.add_argument('--crop', action='store_true', help='whether to crop the video')
parser.add_argument('--crop_factor', type=float, default=1, help='whether to crop the video')
# 反向跟踪
parser.add_argument('--backward', action='store_true', help='whether to backward the tracking')
# 是否可视化支持点
parser.add_argument('--vis_support', action='store_true', help='whether to visualize the support points')
# 查询帧
parser.add_argument('--query_frame', type=int, default=0, help='query frame')
# 设置可视化点的大小
parser.add_argument('--point_size', type=int, default=3, help='point size')
# 是否使用RGBD作为输入
parser.add_argument('--rgbd', action='store_true', help='whether to take the RGBD as input')

parser.add_argument('--start', type=int, help='start process')

parser.add_argument('--end', type=int, help='end process')

parser.add_argument("--fg_video_path", type=str, default=None, help="The video path of the foreground video to be added")

parser.add_argument("--prompt", type=str, default=None, help="The foreground elements that you want to extract from the foreground video")

args = parser.parse_args()

frame_rate = args.fps_vis
grid_size = args.grid_size
model_type = args.model
downsample = args.downsample
root_dir = args.root

def save_mask(mask, frame_idx, output_dir):
    if mask.ndim == 3:
        mask = mask.squeeze()  # 压缩为二维数组
    mask_image = (mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_image)
    mask_image.save(os.path.join(output_dir, f"{frame_idx:05d}.png"))


# 前景整帧输出：和 app.py 一样，只把背景变白，不做 576×576 居中缩放 --------------------------
def create_fg_video(ori_folder, mask_folder, output_video_path, output_video_path1, frame_rate=12):
    """
    ori_folder: 原始前景帧（整帧）
    mask_folder: 对应的二值 mask 帧（前景为 255，背景为 0）
    output_video_path: 保存 mask 视频
    output_video_path1: 保存白底前景 element 视频
    frame_rate: 输出视频帧率
    """

    # 支持的图片后缀
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # 读取 mask 文件列表
    image_files = [f for f in os.listdir(mask_folder)
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # 读取 ori 文件列表
    ori_files = [f for f in os.listdir(ori_folder)
                 if os.path.splitext(f)[1] in valid_extensions]
    ori_files.sort()

    if len(ori_files) != len(image_files):
        print(f"Warning: ori_files ({len(ori_files)}) and mask_files ({len(image_files)}) have different lengths.")
        min_len = min(len(ori_files), len(image_files))
        ori_files = ori_files[:min_len]
        image_files = image_files[:min_len]

    # 用第一帧确定分辨率
    first_mask_path = os.path.join(mask_folder, image_files[0])
    first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        print(f"Error: cannot read first mask frame: {first_mask_path}")
        return 0
    height, width = first_mask.shape[:2]

    # 创建两个视频写入器：mask 视频、白底前景视频，分辨率都与原视频一致
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    video_writer1 = cv2.VideoWriter(output_video_path1, fourcc, frame_rate, (width, height))

    for image_file, ori_file in zip(image_files, ori_files):
        mask_path = os.path.join(mask_folder, image_file)
        ori_path = os.path.join(ori_folder, ori_file)

        # 读 mask（灰度）和原图（BGR）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ori = cv2.imread(ori_path)

        if mask is None or ori is None:
            print(f"Error: cannot read mask or ori frame: {mask_path}, {ori_path}")
            video_writer.release()
            video_writer1.release()
            return 0

        # 尺寸不一致时，强制 resize 到 (width, height)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if ori.shape[:2] != (height, width):
            ori = cv2.resize(ori, (width, height), interpolation=cv2.INTER_LINEAR)

        # 二值化 mask，前景 255，背景 0
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 1) mask 视频：写三通道 BGR
        mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
        video_writer.write(mask_bgr)

        # 2) element 视频：整帧分辨率，mask 外置为白色
        element = ori.copy()
        element[mask_bin == 0] = 255  # 背景变白
        video_writer1.write(element)

    video_writer.release()
    video_writer1.release()
    print(f"Video saved at {output_video_path} and {output_video_path1}")
    return 1
# ---------------------------------------------------------------------------



PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]

# 初始化Grounding DINO模型
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
"""
Step 1: Environment settings and model initialization for SAM 2
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "../ckpts/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

FRAME_THRESHOLD = 49

    

        

start_point = args.start
end_point = args.end
SOURCE_VIDEO_FRAME_DIR = f"../custom_video_frames_{start_point}_{end_point}"
SAVE_TRACKING_RESULTS_DIR = f"../tracking_results_{start_point}_{end_point}"




VIDEO_PATH = args.fg_video_path
elements = args.prompt

# 处理元素列：如果包含多个元素，用逗号分隔
objs = [element.strip() for element in elements.split(',')]

print(f"Processing video: {VIDEO_PATH}")
print(f"Objects to detect: {objs}")

# 以下是原有的代码逻辑，保持不变
video_name = VIDEO_PATH.split('/')[-1]
save_path = os.path.join(args.save_path, video_name)
mask_path = os.path.join(args.mask_path, video_name)
fg_path = os.path.join(args.fg_path, video_name)


TEXT_PROMPT = objs[0] + ", identify the most obvious one only."

# 以下是原有的视频处理逻辑，保持不变
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=0 + FRAME_THRESHOLD)

source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, 
    overwrite=True, 
    image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
        sink.save_image(frame)

# ---------------- 固定为 49 帧的采样策略 ----------------
# 先列出刚刚保存的所有帧
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

num_frames = len(frame_names)
if num_frames == 0:
    raise ValueError(f"No frames found in {SOURCE_VIDEO_FRAME_DIR}")

MIN_FRAMES = 49

if num_frames >= MIN_FRAMES:
    # 已经有不少帧了：从中“均匀”采样 49 帧
    if num_frames < 400:
        indices = np.linspace(0, num_frames - 1, MIN_FRAMES, dtype=int).tolist()
    else:
        indices = np.linspace(0, 400, MIN_FRAMES, dtype=int).tolist()
    selected_names = [frame_names[i] for i in indices]
    selected_set = set(selected_names)
    # 删除未选中的帧文件，使目录里只剩 49 帧
    for name in frame_names:
        if name not in selected_set:
            os.remove(os.path.join(SOURCE_VIDEO_FRAME_DIR, name))
else:
    # 帧数少于 49：先保留原来所有帧，再用 ping-pong 方式补到 49
    selected_names = frame_names.copy()
    needed = MIN_FRAMES - num_frames
    print(f"FG frames too short ({num_frames}), padding to {MIN_FRAMES} by ping-pong repeating frames.")
    # 找到当前最大的编号，新帧就按编号往后接
    existing_indices = [int(os.path.splitext(n)[0]) for n in frame_names]
    max_idx = max(existing_indices) if existing_indices else -1

    n = num_frames
    # 从后往前 [n-1, ..., 0]
    backward_indices = list(range(n - 1, -1, -1))
    # 从前往后 [0, ..., n-1]
    forward_indices = list(range(n))
    # 交替使用：先 backward，再 forward，再 backward...
    patterns = [backward_indices, forward_indices]
    pattern_idx = 0  # 0 -> backward, 1 -> forward

    added = 0
    while added < needed:
        seq = patterns[pattern_idx]
        for idx in seq:
            if added >= needed:
                break
            src_name = frame_names[idx]
            src_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, src_name)
            img = cv2.imread(src_path)
            new_idx = max_idx + 1 + added
            new_name = f"{new_idx:05d}.jpg"
            new_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, new_name)
            cv2.imwrite(new_path, img)
            selected_names.append(new_name)
            added += 1
        # 切换方向：backward <-> forward
        pattern_idx = 1 - pattern_idx


# 重新获取 / 排序一遍剩下的帧名，后面所有逻辑沿用这个 list
frame_names = sorted(
    [p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]],
    key=lambda p: int(os.path.splitext(p)[0])
)

print(f"After resampling, keep exactly {len(frame_names)} frames (should be 49).")
# ---------------- 固定 49 帧结束 ----------------


first_image_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # the frame index we interact with
"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)
inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = grounding_model(**inputs)

# results = processor.post_process_grounded_object_detection(
#     outputs,
#     inputs.input_ids,
#     box_threshold=0.4,
#     text_threshold=0.3,
#     target_sizes=[image.size[::-1]]
# )

results = processor.post_process_grounded_object_detection(
    outputs=outputs,
    threshold=0.4,                     # 统一用 threshold
    target_sizes=[image.size[::-1]],   # (H, W)，你这行是对的
)


input_boxes = results[0]["boxes"].cpu().numpy()
confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]

if len(input_boxes) == 0:
    raise ValueError("No target identity exits")  # 抛出异常并终止程序
elif len(input_boxes) > 1:      # 只选择一个
    input_boxes = [input_boxes[0]]
    class_names = [class_names[0]]

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(np.array(image.convert("RGB")))

# process the detection results
OBJECTS = class_names

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
# convert the mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
# Using box prompt
for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=object_id,
        box=box,
    )


"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
    os.makedirs(SAVE_TRACKING_RESULTS_DIR)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    save_mask(masks, frame_idx, SAVE_TRACKING_RESULTS_DIR)


"""
Step 6: Convert the annotated frames to video
"""
adj = create_fg_video(SOURCE_VIDEO_FRAME_DIR, SAVE_TRACKING_RESULTS_DIR, mask_path, fg_path, frame_rate=frame_rate)
if adj == 0:
    os.remove(mask_path)
    os.remove(fg_path)
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    # video_writer = cv2.VideoWriter(os.path.join(save_path, obj + ".mp4"), fourcc, frame_rate, (width, height))
    video_writer = cv2.VideoWriter(save_path, fourcc, frame_rate, (width, height))
    # write each image to the video
    for image_file in tqdm(frame_names):
        image_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    # source release
    video_writer.release()

print(f"Finished processing video: {VIDEO_PATH}")