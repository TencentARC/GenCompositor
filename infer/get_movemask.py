# 根据拖拽的txt 得到 轨迹移动的mask video -------------------------------------------------------------------------
import cv2
import numpy as np
import argparse
def load_trajectory(trajectory_path):
    """加载轨迹坐标文件"""
    trajectory = []
    with open(trajectory_path, 'r') as f:
        for line in f:
            x, y = map(int, line.strip().split(','))
            trajectory.append((x, y))
    return trajectory

def generate_mask_video_with_trajectory(fg_video_path, source_video_path, output_path, trajectory_path, scale=2.0, below=False):
    # 加载轨迹坐标
    trajectory = load_trajectory(trajectory_path)
    if not trajectory:
        print("Error: Could not load trajectory file")
        return
    
    # 打开视频文件
    fg_cap = cv2.VideoCapture(fg_video_path)
    source_cap = cv2.VideoCapture(source_video_path)
    
    # 获取视频属性
    source_width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = source_cap.get(cv2.CAP_PROP_FPS)
    
    # 确保处理的帧数不超过轨迹点数和视频帧数
    frame_count = min(
        int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(trajectory)
    )
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mask = cv2.VideoWriter(output_path, fourcc, fps, (source_width, source_height), isColor=False)
    
    frame_num = 0
    while frame_num < frame_count:
        ret_fg, fg_frame = fg_cap.read()
        ret_source, source_frame = source_cap.read()
        
        if not ret_fg or not ret_source:
            break
        
        # 获取当前帧的轨迹坐标
        center_x, center_y = trajectory[frame_num]
        
        # 转换为灰度并二值化
        gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros((source_height, source_width), dtype=np.uint8)
        
        if contours:
            # 获取最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 对轮廓进行缩放
            if scale != 1.0:
                M = cv2.moments(largest_contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                scaling_matrix = np.array([[scale, 0, (1-scale)*cx],
                                         [0, scale, (1-scale)*cy]])
                scaled_contour = cv2.transform(largest_contour, scaling_matrix)
                # 更新缩放后的边界框
                x, y, w, h = cv2.boundingRect(scaled_contour)
            
            if not below:
                # 计算在源视频中的位置
                start_x = center_x - w // 2
                start_y = center_y - h // 2
            
                # 处理边界情况
                target_x1 = max(0, start_x)
                target_y1 = max(0, start_y)
                target_x2 = min(source_width, start_x + w)
                target_y2 = min(source_height, start_y + h)
            
                src_x1 = max(0, -start_x)
                src_y1 = max(0, -start_y)
                src_x2 = w - max(0, (start_x + w) - source_width)
                src_y2 = h - max(0, (start_y + h) - source_height)
            else:
                # mask最下方与轨迹一致---------------------------------------------------------
                # 计算在源视频中的位置（底部中点对齐）
                start_x = center_x - w // 2
                start_y = center_y - h  # 关键修改：底部对齐
                # 处理边界情况
                target_x1 = max(0, start_x)
                target_y1 = max(0, start_y)
                target_x2 = min(source_width, start_x + w)
                target_y2 = min(source_height, start_y + h)
                src_x1 = max(0, -start_x)
                src_y1 = max(0, -start_y)
                src_x2 = w - max(0, (start_x + w) - source_width)
                src_y2 = h - max(0, (start_y + h) - source_height)
                # mask最下方与轨迹一致---------------------------------------------------------

            # 轮廓偏移计算
            contour_offset = np.array([target_x1 - src_x1, target_y1 - src_y1])
            
            # 根据是否缩放选择要使用的轮廓
            current_contour = scaled_contour if scale != 1.0 else largest_contour
            shifted_contour = current_contour + contour_offset - np.array([x, y])
            
            # 只填充缩放后的轮廓到最终mask
            cv2.fillPoly(final_mask, [shifted_contour], 255)
        
        # 写入视频
        out_mask.write(final_mask)
        
        frame_num += 1
        if frame_num % 10 == 0:
            print(f"Processing frame {frame_num}/{frame_count}")
    
    # 释放资源
    fg_cap.release()
    source_cap.release()
    out_mask.release()
    print(f"Mask video generated: {output_path}")


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--fg_video_path", type=str, default=None,
                        help="The video path of the foreground video to be added")
    parser.add_argument("--video_path", type=str, default=None,
                        help="The video path of the background video to be edited")
    parser.add_argument("--file_path", type=str, default=None,
                        help="The file path of the trajectory specified by users")
    parser.add_argument("--usr_mask_path", type=str, default=None,
                        help="The video path of the binary video specified by users")
    parser.add_argument("--rescale", type=float, default=1.0,
                        help="The user-specified rescale factor")
    args = parser.parse_args()

    fg_video_path = args.fg_video_path  # 前景视频路径
    source_video_path = args.video_path  # 背景视频路径
    trajectory_path = args.file_path  # 轨迹文件路径
    output_path = args.usr_mask_path  # 输出mask视频路径
    scale = args.rescale  # 设置缩放比例
    
    generate_mask_video_with_trajectory(fg_video_path, source_video_path, output_path, trajectory_path, scale, below = False)