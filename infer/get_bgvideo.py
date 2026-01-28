import cv2
import argparse
import os

def resize_and_save_frames(bg_video_path, save_path, skip_frames=False):
    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"The total number of input video frames: {total_frames}")

    fps_out = 12

    target_width = 720
    target_height = 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps_out, (target_width, target_height))

    #  ---------------------------------
    skip_interval = 0
    if skip_frames:
        if total_frames >= 200 and total_frames < 300:
            skip_interval = 1
        elif total_frames >= 300 and total_frames < 400:
            skip_interval = 2
        elif total_frames >= 400:
            skip_interval = 3
        print(f"Enable frame skipping function, frame skipping interval: {skip_interval}")
    stride = skip_interval + 1
    # ---------------------------------

    frames_written = 0
    frame_idx = 0

    sampled_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0 and frame_idx < 49 * stride:
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)
            sampled_frames.append(resized_frame)
            frames_written += 1
        frame_idx += 1

    MIN_FRAMES = 49

    if frames_written == 0:
        print("Error: No frames were written to output video.")
        cap.release()
        out.release()
        return

    # ---------- Frame interpolation logic：ping-pong ----------
    if frames_written < MIN_FRAMES:
        print(f"Video is short, pad {MIN_FRAMES - frames_written} frames to reach {MIN_FRAMES}.")

        n = len(sampled_frames)
        # The index from back to front, such as [n-1, n-2, ..., 0]
        backward_indices = list(range(n - 1, -1, -1))
        # The index from front to back, such as [0, 1, ..., n-1]
        forward_indices = list(range(n))
        # Alternately ...
        patterns = [backward_indices, forward_indices]
        pattern_idx = 0  # 0 -> backward, 1 -> forward

        while frames_written < MIN_FRAMES:
            seq = patterns[pattern_idx]
            for idx in seq:
                if frames_written >= MIN_FRAMES:
                    break
                frame_to_add = sampled_frames[idx]
                out.write(frame_to_add)
                frames_written += 1
            # Switch direction
            pattern_idx = 1 - pattern_idx
    # ----------------------------------------------------------

    cap.release()
    out.release()
    print(f"Video saved to {save_path}, total output frames: {frames_written}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_video_path', type=str, required=True, help='Path to the input MP4 file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--skip_frames', action='store_true',
                        help='Enable frame skipping based on total frame count')

    args = parser.parse_args()

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    resize_and_save_frames(args.bg_video_path, args.save_path, args.skip_frames)
