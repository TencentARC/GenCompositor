import cv2
import numpy as np
import argparse

def draw_and_save_trajectory(source_video_path, trajectory_txt_path):
    cap = cv2.VideoCapture(source_video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频文件")
        return None


    # # 新增：获取原始尺寸并计算半尺寸
    # original_height, original_width = first_frame.shape[:2]
    # window_width = original_width // 2
    # window_height = original_height // 2


    trajectory_points = []
    drawing = False

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            trajectory_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            trajectory_points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    # 创建窗口并绘制
    cv2.namedWindow("Draw Trajectory (Press ESC when done)")
    cv2.setMouseCallback("Draw Trajectory (Press ESC when done)", mouse_callback)

    while True:
        display_frame = first_frame.copy()
        for i in range(1, len(trajectory_points)):
            cv2.line(display_frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)
        for point in trajectory_points:
            cv2.circle(display_frame, point, 1, (0, 0, 255), -1)
            # cv2.circle(display_frame, point, 3, (0, 0, 255), -1)
        cv2.imshow("Draw Trajectory (Press ESC when done)", display_frame)

        if cv2.waitKey(1) == 27:  # ESC键
            # 保存带有轨迹的第一帧图片
            output_image_path = trajectory_txt_path.replace('.txt', '.png')  # 使用相同路径但扩展名为.png
            cv2.imwrite(output_image_path, display_frame)
            print(f"The first frame with the specified track has been saved to: {output_image_path}")
            break

    cv2.destroyAllWindows()

    if not trajectory_points:
        print("No track is drawn")
        return None

    # 情况0：只有1个点 → 使用光流生成轨迹
    if len(trajectory_points) == 1:
        print("Detected a single point, optical flow will be used to generate a trajectory...")

        # 准备光流计算
        prev_frame = first_frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_point = np.array([[trajectory_points[0]]], dtype=np.float32)

        # 初始化轨迹点列表（第一个点已存在）
        flow_points = [tuple(trajectory_points[0])]

        for _ in range(total_frames - 1):
            ret, next_frame = cap.read()
            if not ret:
                break

            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            # 计算光流
            next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_point, None
            )

            # 更新轨迹点
            if status[0] == 1:  # 跟踪成功
                print('success')
                new_point = tuple(map(int, next_point[0][0]))
                flow_points.append(new_point)
                prev_point = next_point
                prev_gray = next_gray
            else:
                # 跟踪失败，复制上一个点
                flow_points.append(flow_points[-1])

        cap.release()
        trajectory_points = flow_points[:total_frames]  # 确保长度匹配

    # 情况1：点数 < 帧数 → 线性插值
    elif len(trajectory_points) < total_frames:
        if len(trajectory_points) < 2:
            print("At least 2 points are required !")
            return None

        interpolated_points = []
        segments = len(trajectory_points) - 1
        points_per_segment = total_frames // segments

        for i in range(segments):
            start_point = trajectory_points[i]
            end_point = trajectory_points[i + 1]

            # 当前线段的插值点数（最后一个线段处理剩余帧）
            n_points = points_per_segment if i < segments - 1 else (total_frames - (segments - 1) * points_per_segment)

            for j in range(n_points):
                ratio = j / max(1, n_points - 1)  # 避免除以零
                x = int(start_point[0] + ratio * (end_point[0] - start_point[0]))
                y = int(start_point[1] + ratio * (end_point[1] - start_point[1]))
                interpolated_points.append((x, y))

        # 确保点数严格匹配（处理整数除法误差）
        interpolated_points = interpolated_points[:total_frames]
        trajectory_points = interpolated_points

    # 情况2：点数 > 帧数 → 均匀抽点
    elif len(trajectory_points) > total_frames:
        step = len(trajectory_points) / total_frames
        sampled_points = []
        for i in range(total_frames):
            idx = min(int(i * step), len(trajectory_points) - 1)
            sampled_points.append(trajectory_points[idx])
        trajectory_points = sampled_points

    # 保存轨迹到TXT文件
    with open(trajectory_txt_path, 'w') as f:
        for point in trajectory_points:
            f.write(f"{point[0]},{point[1]}\n")

    print(f"Track information has been saved to: {trajectory_txt_path}")
    print(f"Total video frames: {total_frames}, Final track points: {len(trajectory_points)}")
    return trajectory_points

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

    fg_video_path = args.fg_video_path  # The path of foreground video
    source_video_path = args.video_path  # The path of background video
    trajectory_path = args.file_path  # The path of trajectory file
    output_path = args.usr_mask_path  # The path of output mask video
    scale = args.rescale  # Specify rescale factor and produce final binary mask of foreground element

    draw_and_save_trajectory(source_video_path, trajectory_path)

    generate_mask_video_with_trajectory(fg_video_path, source_video_path, output_path, trajectory_path, scale, below = True)
