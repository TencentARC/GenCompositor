import cv2
import argparse
import os

def resize_and_save_frames(bg_video_path, save_path, skip_frames=False):
    # 打开视频文件
    cap = cv2.VideoCapture(bg_video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # 获取原视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"输入视频总帧数: {total_frames}")
    
    # 获取原视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 定义目标分辨率 (宽x高)
    target_width = 720
    target_height = 480
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (target_width, target_height))
    
    frame_count = 0
    saved_count = 0
    
    # 计算跳帧间隔（如果启用跳帧功能）
    skip_interval = 0
    if skip_frames:
        if total_frames >= 200 and total_frames < 300:
            skip_interval = 1
        elif total_frames >= 300 and total_frames < 400:
            skip_interval = 2
        elif total_frames >= 400:
            # 对于400帧以上的视频，每增加100帧，跳帧间隔增加1
            skip_interval = (total_frames // 100) - 2
        print(f"启用跳帧功能，跳帧间隔: {skip_interval}")
    
    while saved_count < 100:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # 如果启用了跳帧功能，并且当前帧不是需要保存的帧，则跳过
        if skip_frames and frame_count % (skip_interval + 1) != 0:
            frame_count += 1
            continue
        
        # 调整帧分辨率
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        # 写入调整后的帧
        out.write(resized_frame)
        
        saved_count += 1
        frame_count += 1
        print(f"Processed frame {saved_count}/100 (原始帧: {frame_count})")
    
    # 释放资源
    cap.release()
    out.release()
    print(f"Video saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_video_path', type=str, required=True, help='Path to the input MP4 file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--skip_frames', action='store_true', help='Enable frame skipping based on total frame count')
    
    args = parser.parse_args()
    
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    resize_and_save_frames(args.bg_video_path, args.save_path, args.skip_frames)