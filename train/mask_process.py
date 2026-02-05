import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import math

def generate_random_brush(h, w):
    """生成随机笔画mask"""
    mask = Image.new('L', (w, h), 0)
    average_radius = math.sqrt(h*h+w*w) / 8
    max_tries = 5
    min_num_vertex, max_num_vertex = 1, 8
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width, max_width = 128, 256
    # min_width = min(h, w) // 5  # 使用图像最大边长的1/5作为笔画宽度
    # max_width = min_width
    
    num_tries = np.random.choice([_ for _ in range(max_tries)], p=[0.05, 0.3, 0.3, 0.3, 0.05])
    for _ in range(num_tries):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                        v[1] - width//2,
                        v[0] + width//2,
                        v[1] + width//2),
                        fill=1)
    
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def transform_video_masks(
    video_masks, 
    p_brush=0.25,
    p_rect=0.25,
    p_ellipse=0.2, 
    p_circle=0.2,
    p_random_brush=0.1,
    margin_ratio=0.1,
    shape_scale_min=1.1,
    shape_scale_max=1.5,
    brush_iterations=1
):
    """
    转换视频序列中的masks，整个视频序列使���相同的变换参数
    
    参数:
    video_masks: numpy array, shape (F, H, W, C)
    """
    F, H, W, C = video_masks.shape
    transformed_video_masks = np.zeros_like(video_masks)
    
    # 为整个视频序列选择一种mask变化模式
    choice = np.random.choice(
        ['brush', 'rect', 'ellipse', 'circle', 'random_brush'], 
        p=[p_brush, p_rect, p_ellipse, p_circle, p_random_brush]
    )
    
    # 预先确定所有随机参数
    if choice == 'brush':
        # 选择一种形态学操作类型
        morph_type = np.random.choice([
            'dilate_erode',  # 闭运算
            'erode_dilate',  # 开运算
            'dilate_only',   # 仅膨胀
            'combined'       # 组合操作
        ])
        # 决定是否使用高斯模糊
        use_blur = np.random.random() < 0.1
        # print(f"choice: {choice}, morph_type: {morph_type}, use_blur: {use_blur}")
    
    elif choice == 'random_brush':
        # 为整个序列生成一个随机笔画mask
        first_frame_brush = generate_random_brush(H, W)
        # print(f"choice: {choice}")

    elif choice == 'rect':
        # 预定义矩形参数
        rect_angle = np.random.uniform(0, 360)
        width_scale = np.random.uniform(shape_scale_min, shape_scale_max)
        height_scale = np.random.uniform(shape_scale_min, shape_scale_max)
        
    elif choice == 'ellipse':
        # 预定义椭圆参数
        width_scale = np.random.uniform(shape_scale_min/2, shape_scale_max/2)
        height_scale = np.random.uniform(shape_scale_min/2, shape_scale_max/2)
        angle = np.random.uniform(0, 360)
        
    else:  # circle
        # 预定义圆形参数
        radius_scale = np.random.uniform(shape_scale_min/2, shape_scale_max/2)
    
    # 为rect、ellipse和circle模式预先生成mask
    if choice in ['rect', 'ellipse', 'circle']:
        # 使用第一帧来获取边界框
        first_frame = video_masks[0]
        H, W, C = first_frame.shape
        y_indices, x_indices = np.where(first_frame[:,:,0] > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return video_masks
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 添加边界扰动
        margin = int(min(H, W) * margin_ratio)
        x_min = max(0, x_min - np.random.randint(0, margin))
        x_max = min(W, x_max + np.random.randint(0, margin))
        y_min = max(0, y_min - np.random.randint(0, margin))
        y_max = min(H, y_max + np.random.randint(0, margin))
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        width = x_max - x_min
        height = y_max - y_min
        
        # 预先生成形状mask
        first_frame_shape = np.zeros((H, W), dtype=np.uint8)
        
        if choice == 'rect':
            rect = (
                (float(center_x), float(center_y)),
                (float(width * width_scale), float(height * height_scale)),
                float(rect_angle)
            )
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(first_frame_shape, [box], 1)
            
        elif choice == 'ellipse':
            axes_length = (int(width * width_scale), int(height * height_scale))
            cv2.ellipse(first_frame_shape,
                       (center_x, center_y),
                       axes_length,
                       angle,
                       0, 360, 1, -1)
            
        else:  # circle
            radius = int(max(width, height) * radius_scale)
            cv2.circle(first_frame_shape,
                      (center_x, center_y),
                      radius,
                      1, -1)

    def transform_frame(mask):
        H, W, C = mask.shape
        transformed_mask = np.zeros((H, W, C), dtype=np.uint8)
        
        if choice == 'random_brush':
            transformed_mask[:,:,0] = first_frame_brush
        elif choice in ['rect', 'ellipse', 'circle']:
            transformed_mask[:,:,0] = first_frame_shape
        elif choice == 'brush':
            kernel = np.ones((32, 32), np.uint8)
            
            if morph_type == 'dilate_erode':
                dilated = cv2.dilate(mask[:,:,0].astype(np.uint8), kernel, iterations=brush_iterations)
                transformed_mask[:,:,0] = cv2.erode(dilated, kernel, iterations=brush_iterations)
                
            elif morph_type == 'erode_dilate':
                eroded = cv2.erode(mask[:,:,0].astype(np.uint8), kernel, iterations=brush_iterations)
                transformed_mask[:,:,0] = cv2.dilate(eroded, kernel, iterations=brush_iterations)
                
            elif morph_type == 'dilate_only':
                transformed_mask[:,:,0] = cv2.dilate(mask[:,:,0].astype(np.uint8), kernel, iterations=brush_iterations)
                
            else:  # combined
                eroded = cv2.erode(mask[:,:,0].astype(np.uint8), kernel, iterations=brush_iterations)
                opened = cv2.dilate(eroded, kernel, iterations=brush_iterations)
                dilated = cv2.dilate(opened, kernel, iterations=brush_iterations)
                transformed_mask[:,:,0] = cv2.erode(dilated, kernel, iterations=brush_iterations)
                
            if use_blur:
                blur_size = max(3, 8 // 4)
                if blur_size % 2 == 0:
                    blur_size += 1
                transformed_mask[:,:,0] = cv2.GaussianBlur(transformed_mask[:,:,0], (blur_size, blur_size), 0)
                transformed_mask = (transformed_mask > 0.5).astype(np.uint8)
            
        # 复制到其他通道
        transformed_mask[:,:,1:] = transformed_mask[:,:,0:1]
        return transformed_mask
    
    # 处理每一帧
    for f in range(F):
        transformed_video_masks[f] = transform_frame(video_masks[f])
            
    return transformed_video_masks

def test_transform_video_masks():
    """测试mask变换函数"""
    # 创建测试用的视频mask序列
    F, H, W, C = 10, 480, 720, 3
    video_masks = np.zeros((F, H, W, C), dtype=np.uint8)
    
    # 创建一些模拟的分割mask
    for f in range(F):
        # 创建基础mask
        frame_mask = np.zeros((H, W), dtype=np.uint8)
        
        # 定义缩放因子，使mask更大
        scale = min(H, W) // 2  # 根据图像尺寸自适应缩放
        
        # 添加一个不规则形状（模拟人物轮廓）
        points = np.array([
            [scale * (0.5 + np.random.randint(-5, 5)/100), scale * (0.3 + np.random.randint(-2, 2)/100)],  # 头部
            [scale * (0.3 + np.random.randint(-3, 3)/100), scale * (0.5 + np.random.randint(-2, 2)/100)],  # 左肩
            [scale * (0.7 + np.random.randint(-3, 3)/100), scale * (0.5 + np.random.randint(-2, 2)/100)],  # 右肩
            [scale * (0.2 + np.random.randint(-3, 3)/100), scale * (0.9 + np.random.randint(-2, 2)/100)],  # 左脚
            [scale * (0.8 + np.random.randint(-3, 3)/100), scale * (0.9 + np.random.randint(-2, 2)/100)],  # 右脚
        ], dtype=np.int32)
        
        # 将轮廓移动到图像中心
        center_offset_x = W // 2 - scale // 2
        center_offset_y = H // 2 - scale // 2
        points[:, 0] += center_offset_x
        points[:, 1] += center_offset_y
        
        # 绘制人物轮廓
        cv2.fillPoly(frame_mask, [points], 1)
        
        # 添加更大的随机物体
        for _ in range(np.random.randint(2, 4)):
            cx = np.random.randint(10, W-10)
            cy = np.random.randint(10, H-10)
            radius = np.random.randint(scale//20, scale//10)  # 增大随机物体的半径
            cv2.circle(frame_mask, (cx, cy), radius, 1, -1)
        
        # 应用一些随机变形和模糊，使mask更自然
        if np.random.random() < 0.5:
            kernel = np.ones((3, 3), np.uint8)
            frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel)
        
        # 复制到所有通道
        video_masks[f, :, :, :] = frame_mask[:, :, np.newaxis]
    
    # 测试不同的变换参数组合
    test_cases = [
        # 测试全部使用brush变换
        {'p_brush': 1.0, 'p_rect': 0, 'p_ellipse': 0, 'p_circle': 0, 'p_random_brush': 0, 'brush_iterations': 2},
        # 测试全部使用矩形变换
        {'p_brush': 0, 'p_rect': 1.0, 'p_ellipse': 0, 'p_circle': 0, 'p_random_brush': 0},
        # 测试全部使用椭圆变换
        {'p_brush': 0, 'p_rect': 0, 'p_ellipse': 1.0, 'p_circle': 0, 'p_random_brush': 0},
        # 测试全部使用圆形变换
        {'p_brush': 0, 'p_rect': 0, 'p_ellipse': 0, 'p_circle': 1.0, 'p_random_brush': 0},
        # 测试全部使用随机笔画变换
        {'p_brush': 0, 'p_rect': 0, 'p_ellipse': 0, 'p_circle': 0, 'p_random_brush': 1.0},
        # 测试均匀分布（包含随机笔画）
        {'p_brush': 0.2, 'p_rect': 0.2, 'p_ellipse': 0.2, 'p_circle': 0.2, 'p_random_brush': 0.2},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n测试用例 {i + 1}:")
        print(f"参数: {params}")
        
        transformed_masks = transform_video_masks(
            video_masks,
            **params
        )
        
        # 基本断言测试
        assert transformed_masks.shape == video_masks.shape, "转换后的mask形状应该保持不变"
        assert transformed_masks.dtype == video_masks.dtype, "转换后的mask数据类型应该保持不变"
        assert np.any(transformed_masks != video_masks), "转换后的mask应该与原始mask不同"
        
        # 保存结果
        save_test_results(video_masks, transformed_masks, i)

def save_test_results(original_masks, transformed_masks, test_case_index):
    """保存测试结果为图片"""
    import matplotlib.pyplot as plt
    
    # 创建一个图像网格来显示原始mask和转换后的mask
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # 显示5帧原始mask
    for i in range(5):
        axes[0, i].imshow(original_masks[i, :, :, 0], cmap='gray')
        axes[0, i].set_title(f'Original Frame {i}')
        axes[0, i].axis('off')
    
    # 显示5帧转换后的mask
    for i in range(5):
        axes[1, i].imshow(transformed_masks[i, :, :, 0], cmap='gray')
        axes[1, i].set_title(f'Transformed Frame {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'mask_transform_test_{test_case_index}.png')
    plt.close()

if __name__ == "__main__":
    test_transform_video_masks()
    print("所有测试完成！")