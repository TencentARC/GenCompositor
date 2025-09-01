import cv2
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# def create_video_from_images(image_folder, output_video_path, frame_rate=24):
#     # define valid extension
#     valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
#     # get all image files in the folder
#     image_files = [f for f in os.listdir(image_folder) 
#                    if os.path.splitext(f)[1] in valid_extensions]
#     image_files.sort()  # sort the files in alphabetical order
#     print(image_files)
#     if not image_files:
#         raise ValueError("No valid image files found in the specified folder.")
    
#     # load the first image to get the dimensions of the video
#     first_image_path = os.path.join(image_folder, image_files[0])
#     first_image = cv2.imread(first_image_path)
#     height, width, _ = first_image.shape
    
#     # create a video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
#     # write each image to the video
#     for image_file in tqdm(image_files):
#         image_path = os.path.join(image_folder, image_file)
#         image = cv2.imread(image_path)
#         video_writer.write(image)
    
#     # source release
#     video_writer.release()
#     print(f"Video saved at {output_video_path}")





output_height = 576
output_width = 576
# output_height = 720
# output_width = 1280
threshold = 0.8
kernel = np.ones((9, 9), np.uint8)  # 定义形态学操作的内核大小


def process_frame(image_path, ori_path, height, width):
    image = cv2.imread(image_path)
    ori = cv2.imread(ori_path)

    # 将mask二值化（确保mask是二元的）
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # 1. 使用形态学操作去除零星噪声
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # 开运算去除噪声
        
    # 检测是否有目标物体
    rows, cols, _ = np.where(image == 255)
    if len(rows) == 0 or len(cols) == 0:
        print(f"Warning: No object found in mask for {os.path.basename(ori_path)}")
        return None

    # 计算白色区域的 bounding box
    min_x, min_y = min(cols), min(rows)
    max_x, max_y = max(cols), max(rows)
    object_width = max_x - min_x
    object_height = max_y - min_y

    # 检查白色区域是否接触图像边缘
    touches_edge = (0 in rows or (height - 1) in rows or 0 in cols or (width - 1) in cols)

    return (image, ori, object_width, object_height, touches_edge)


def calculate_max_mask_ratio(mask_folder, output_width, output_height):
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    image_files = [f for f in os.listdir(mask_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()

    max_ratio = 0

    for image_file in image_files:
        image_path = os.path.join(mask_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        rows, cols = np.where(image == 255)
        if len(rows) == 0 or len(cols) == 0:
            continue

        min_x, min_y = min(cols), min(rows)
        max_x, max_y = max(cols), max(rows)
        object_width = max_x - min_x
        object_height = max_y - min_y

        width_ratio = object_width / output_width
        height_ratio = object_height / output_height
        current_max_ratio = max(width_ratio, height_ratio)

        if current_max_ratio > max_ratio:
            max_ratio = current_max_ratio

    return max_ratio

def create_fg_video(ori_folder, mask_folder, output_video_path, output_video_path1, frame_rate=24):
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

    # 如果前12帧中所有白色区域都接触边缘
    if min_object_width == float('inf') or min_object_height == float('inf'):
        print("Warning: All objects in the first 12 frames touch the image edge.")
        video_writer.release()
        video_writer1.release()
        return 0  # 终止当前函数运行

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

        # 检查白色区域是否接触图像边缘
        if (0 in rows or (height - 1) in rows or 0 in cols or (width - 1) in cols):
            # 计算当前 bounding box 面积与最小 bounding box 面积的比值
            current_area = object_width * object_height
            min_area = min_object_width * min_object_height
            ratio = current_area / min_area

            # 如果比值小于阈值
            if ratio < threshold:
                print(f"Warning: Object in mask {image_file} touches the image edge and is too small (ratio: {ratio}).")
                video_writer.release()
                video_writer1.release()
                return 0  # 终止当前函数运行

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