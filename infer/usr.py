import cv2
import numpy as np
import argparse

def draw_and_save_trajectory(source_video_path, trajectory_txt_path):
    """在视频第一帧上绘制轨迹，自动调整点数匹配帧数（插值或抽点）"""
    cap = cv2.VideoCapture(source_video_path)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频文件")
        return None


    # # 新增：获取原始尺寸并计算半尺寸
    # original_height, original_width = first_frame.shape[:2]
    # window_width = original_width // 2
    # window_height = original_height // 2


    # 初始化轨迹点列表
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
    # # 修改窗口创建方式
    # cv2.namedWindow("Draw Trajectory (Press ESC when done)", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Draw Trajectory (Press ESC when done)", window_width, window_height)
    # cv2.setMouseCallback("Draw Trajectory (Press ESC when done)", mouse_callback)

    while True:
        display_frame = first_frame.copy()
        for i in range(1, len(trajectory_points)):
            cv2.line(display_frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)
        for point in trajectory_points:
            cv2.circle(display_frame, point, 3, (0, 255, 0), -1)
            # cv2.circle(display_frame, point, 3, (0, 0, 255), -1)
        cv2.imshow("Draw Trajectory (Press ESC when done)", display_frame)

        if cv2.waitKey(1) == 27:  # ESC键
            # 保存带有轨迹的第一帧图片
            output_image_path = trajectory_txt_path.replace('.txt', '.png')  # 使用相同路径但扩展名为.png
            cv2.imwrite(output_image_path, display_frame)
            print(f"已保存带有轨迹的第一帧图片到: {output_image_path}")
            break

    cv2.destroyAllWindows()

    if not trajectory_points:
        print("没有绘制轨迹")
        return None

    # 情况0：只有1个点 → 使用光流生成轨迹
    if len(trajectory_points) == 1:
        print("检测到单点，将使用光流生成轨迹...")

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
            print("至少需要2个点才能插值")
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

    print(f"轨迹已保存到: {trajectory_txt_path}")
    print(f"视频总帧数: {total_frames}, 最终轨迹点数: {len(trajectory_points)}")
    return trajectory_points


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--video_path", type=str, default=None, help="The video path of the background video to be edited")
    parser.add_argument("--file_path", type=str, default=None,
                        help="The file path of the trajectory specified by users")
    args = parser.parse_args()

    source_video = args.video_path
    trajectory_txt = args.file_path
    draw_and_save_trajectory(source_video, trajectory_txt)