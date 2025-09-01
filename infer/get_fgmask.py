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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.video_utils import calculate_max_mask_ratio, process_frame
from concurrent.futures import ThreadPoolExecutor, as_completed

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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
parser.add_argument('--fps_vis', type=int, default=24, help='len_track')
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


# 前景居中-------------------------------------------------------------------------
def create_fg_video(ori_folder, mask_folder, output_video_path, output_video_path1, frame_rate=24):
    output_height = 576
    output_width = 576
    threshold = 0.8
    kernel = np.ones((9, 9), np.uint8)  # 定义形态学操作的内核大小
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(mask_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    ori_files = [f for f in os.listdir(ori_folder) 
                 if os.path.splitext(f)[1] in valid_extensions]
    ori_files.sort()  # sort the files in alphabetical order

    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(mask_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    video_writer1 = cv2.VideoWriter(output_video_path1, fourcc, frame_rate, (output_width, output_height))

    # 计算所有帧的最大 mask_ratio
    max_mask_ratio = calculate_max_mask_ratio(mask_folder, output_width, output_height)

    de_scale = int(max_mask_ratio) + 1

    # 初始化变量
    min_object_width = float('inf')  # 记录最小 bounding box 宽度
    min_object_height = float('inf')  # 记录最小 bounding box 高度
    frame_count = 0  # 当前帧数
    ups2 = 0

    # 使用多线程处理前12帧
    with ThreadPoolExecutor() as executor:
        futures = []
        for image_file, ori_file in zip(image_files[:12], ori_files[:12]):
            image_path = os.path.join(mask_folder, image_file)
            ori_path = os.path.join(ori_folder, ori_file)
            futures.append(executor.submit(process_frame, image_path, ori_path, height, width))

        for future in futures:
            result = future.result()
            if result is None:
                video_writer.release()
                video_writer1.release()
                return 0

            image, ori, object_width, object_height, touches_edge = result

            if touches_edge:
                print(f"Warning: Object in mask {image_file} touches the image edge.")
            else:
                # 更新最小 bounding box
                min_object_width = min(min_object_width, object_width)
                min_object_height = min(min_object_height, object_height)

            # 将前12帧写入视频
            video_writer.write(image)
            # 处理前12帧的背景替换
            white_background = np.ones_like(ori) * 255  # 创建白色背景
            masked_frame = cv2.bitwise_and(ori, image)  # 保留物体区域
            inverted_mask = cv2.bitwise_not(image)  # 反转mask
            white_background = cv2.bitwise_and(white_background, inverted_mask)  # 保留白色背景
            processed_frame = cv2.add(masked_frame, white_background)  # 结合物体和白色背景

            # 找到mask中白色区域的边界
            rows, cols, _ = np.where(image == 255)
            min_x, min_y = min(cols), min(rows)
            max_x, max_y = max(cols), max(rows)
            object_width = max_x - min_x
            object_height = max_y - min_y

            # 裁剪物体区域
            cropped_object = processed_frame[min_y:max_y, min_x:max_x]
            # 创建一个固定大小的白色背景画布
            canvas = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

            # 计算物体尺寸与画布尺寸的百分比
            width_ratio = object_width / output_width
            height_ratio = object_height / output_height
            max_ratio = max(width_ratio, height_ratio)  # 取宽度和高度的最大比例

            if frame_count == 0 and max_mask_ratio < 1:
                if max_ratio < 0.1:        # 如果第0帧的物体尺寸小于画布尺寸的10%
                    ups2 = 4
                elif max_ratio < 0.3:
                    ups2 = 2

            # 调整物体大小
            if ups2:
                if ups2 == 4 and max_ratio >=0.2:
                    video_writer.release()
                    video_writer1.release()
                    return 0  # 终止当前函数运行
                elif ups2 ==2 and max_ratio >= 0.4:
                    video_writer.release()
                    video_writer1.release()
                    return 0  # 终止当前函数运行
                # 放大一倍
                new_width = int(object_width * ups2)
                new_height = int(object_height * ups2)
                cropped_object = cv2.resize(cropped_object, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                object_width = new_width
                object_height = new_height
            else:
                new_width = int(object_width // de_scale)
                new_height = int(object_height // de_scale)
                cropped_object = cv2.resize(cropped_object, (new_width, new_height), interpolation=cv2.INTER_AREA)
                object_width = new_width
                object_height = new_height

            # 计算物体在画布上的起始位置（居中）
            start_x = (output_width - object_width) // 2
            start_y = (output_height - object_height) // 2

            # 将裁剪后的物体放置到画布上
            canvas[start_y:start_y + object_height, start_x:start_x + object_width] = cropped_object[:object_height, :object_width]
            video_writer1.write(canvas)

            frame_count += 1

    # 处理剩余的帧
    for image_file, ori_file in zip(image_files[12:], ori_files[12:]):
        image_path = os.path.join(mask_folder, image_file)
        ori_path = os.path.join(ori_folder, ori_file)
        image = cv2.imread(image_path)
        ori = cv2.imread(ori_path)

        # 将mask二值化（确保mask是二元的）
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # 1. 使用形态学操作去除零星噪声
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # 开运算去除噪声

        # 检测是否有目标物体
        rows, cols, _ = np.where(image == 255)
        if len(rows) == 0 or len(cols) == 0:
            print(f"Warning: No object found in mask for {ori_file}")
            video_writer.release()
            video_writer1.release()
            return 0

        # 计算白色区域的 bounding box
        min_x, min_y = min(cols), min(rows)
        max_x, max_y = max(cols), max(rows)
        object_width = max_x - min_x
        object_height = max_y - min_y

        # 屏蔽背景：保留物体区域，背景替换为白色
        white_background = np.ones_like(ori) * 255  # 创建白色背景
        # print('ori: ', ori.shape, ori.dtype, '|', 'image: ', image.shape, image.dtype)
        masked_frame = cv2.bitwise_and(ori, image)  # 保留物体区域
        inverted_mask = cv2.bitwise_not(image)  # 反转mask
        white_background = cv2.bitwise_and(white_background, inverted_mask)  # 保留白色背景
        processed_frame = cv2.add(masked_frame, white_background)  # 结合物体和白色背景

        # 找到mask中白色区域的边界
        min_x, min_y = min(cols), min(rows)
        max_x, max_y = max(cols), max(rows)
        object_width = max_x - min_x
        object_height = max_y - min_y

        # 裁剪物体区域
        cropped_object = processed_frame[min_y:max_y, min_x:max_x]
        # 创建一个固定大小的白色背景画布
        canvas = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        # 计算物体尺寸与画布尺寸的百分比
        width_ratio = object_width / output_width
        height_ratio = object_height / output_height
        max_ratio = max(width_ratio, height_ratio)  # 取宽度和高度的最大比例

        # 调整物体大小
        if ups2:
            if ups2 == 4 and max_ratio >=0.2:
                video_writer.release()
                video_writer1.release()
                return 0  # 终止当前函数运行
            elif ups2 ==2 and max_ratio >= 0.4:
                video_writer.release()
                video_writer1.release()
                return 0  # 终止当前函数运行
            # 放大一倍
            new_width = int(object_width * ups2)
            new_height = int(object_height * ups2)
            cropped_object = cv2.resize(cropped_object, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            object_width = new_width
            object_height = new_height
        else:
            new_width = int(object_width // de_scale)
            new_height = int(object_height // de_scale)
            cropped_object = cv2.resize(cropped_object, (new_width, new_height), interpolation=cv2.INTER_AREA)
            object_width = new_width
            object_height = new_height

        # 计算物体在画布上的起始位置（居中）
        start_x = (output_width - object_width) // 2
        start_y = (output_height - object_height) // 2
        # 将裁剪后的物体放置到画布上
        canvas[start_y:start_y + object_height, start_x:start_x + object_width] = cropped_object[:object_height, :object_width]
        
        video_writer.write(image)
        video_writer1.write(canvas)
    
    # source release
    video_writer.release()
    video_writer1.release()
    print(f"Video saved at {output_video_path}")
    return 1


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

FRAME_THRESHOLD = 50

    

        

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
        # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        frame = cv2.resize(frame, (720, 480))
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

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

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
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