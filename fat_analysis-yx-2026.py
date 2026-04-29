import cv2
import numpy as np
from skimage import morphology, filters, exposure
import os
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 单位换算常量
PIXEL_TO_UM2 = 0.0358  # 每像素对应的平方微米（修正为0.0358）

# 面积区间（平方微米）对应的像素阈值
AREA_THRESHOLDS = {
    '0-1um²': (0, int(1 / PIXEL_TO_UM2)),                  # 0 - 28 像素
    '1-5um²': (int(1 / PIXEL_TO_UM2), int(5 / PIXEL_TO_UM2)),        # 28 - 140 像素
    '5-15um²': (int(5 / PIXEL_TO_UM2), int(15 / PIXEL_TO_UM2)),      # 140 - 419 像素
    '15-50um²': (int(15 / PIXEL_TO_UM2), int(50 / PIXEL_TO_UM2)),    # 419 - 1397 像素
    '50-200um²': (int(50 / PIXEL_TO_UM2), int(200 / PIXEL_TO_UM2)),  # 1397 - 5587 像素
    '200-500um²': (int(200 / PIXEL_TO_UM2), int(500 / PIXEL_TO_UM2)), # 5587 - 13966 像素
    '500+um²': (int(500 / PIXEL_TO_UM2), float('inf'))               # 13966+ 像素
}

# 科学参数配置
CONFIG = {
    'min_fat_area': 1,      # 最小脂肪粒面积(像素)
    'max_fat_area': 50000,    # 最大脂肪粒面积(像素)
    'gaussian_kernel': (7,7), # 高斯模糊核大小
    'opening_kernel': 5,     # 开运算核大小
    'clahe_clip': 1.5,       # CLAHE对比度限制
    'clahe_grid': (16,16),   # CLAHE网格大小
    'debug': True            # 是否输出调试图像
}

def setup_logging(output_dir):
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s -  %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'fat_analysis.log')),
            logging.StreamHandler()
        ]
    )

def get_area_category(area_pixels):
    """根据像素面积返回对应的平方微米区间类别"""
    for category, (min_val, max_val) in AREA_THRESHOLDS.items():
        if min_val <= area_pixels < max_val:
            return category
    return '500+um²'  # 默认返回最大区间

def process_image(img_path, output_dir):
    """处理单张脂肪细胞图像"""
    try:
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("无法读取图像文件")
            
        # 创建调试目录
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        filename = os.path.basename(img_path)
        
        # 1. 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG['clahe_clip'],
            tileGridSize=CONFIG['clahe_grid'])
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, CONFIG['gaussian_kernel'], 0)
        
        if CONFIG['debug']:
            plt.imsave(os.path.join(debug_dir, f'1_preprocess_{filename}'), blurred, cmap='gray')

        # 2. 自适应阈值分割
        thresh_value = filters.threshold_otsu(blurred)
        binary = (blurred > thresh_value).astype(np.uint8) * 255
        
        if CONFIG['debug']:
            plt.imsave(os.path.join(debug_dir, f'2_binary_{filename}'), binary, cmap='gray')

        # 3. 形态学处理
        kernel = morphology.square(CONFIG['opening_kernel'])
        cleaned = morphology.opening(binary, kernel)
        
        if CONFIG['debug']:
            plt.imsave(os.path.join(debug_dir, f'3_cleaned_{filename}'), cleaned, cmap='gray')

        # 4. 轮廓检测与分析
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fat_count = 0
        total_area = 0
        valid_contours = []
        
        # 面积分布统计（按平方微米区间）
        area_distribution = {
            '0-1um²': 0,
            '1-5um²': 0,
            '5-15um²': 0,
            '15-50um²': 0,
            '50-200um²': 0,
            '200-500um²': 0,
            '500+um²': 0
        }
        
        # 各面积区间的面积总和（像素）
        area_sum_distribution = {
            '0-1um²': 0,
            '1-5um²': 0,
            '5-15um²': 0,
            '15-50um²': 0,
            '50-200um²': 0,
            '200-500um²': 0,
            '500+um²': 0
        }
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if CONFIG['min_fat_area'] <= area <= CONFIG['max_fat_area']:
                fat_count += 1
                total_area += area
                valid_contours.append(cnt)
                
                # 获取面积所属区间并累加
                category = get_area_category(area)
                area_distribution[category] += 1
                area_sum_distribution[category] += area
        
        # 5. 结果可视化
        result_img = image.copy()
        cv2.drawContours(result_img, valid_contours, -1, (0,255,0), 1)
        
        # 保存结果
        output_path = os.path.join(output_dir, 'processed', f'result_{filename}')
        cv2.imwrite(output_path, result_img)
        
        if CONFIG['debug']:
            plt.imsave(os.path.join(debug_dir, f'4_result_{filename}'), cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        
        # 计算各面积区间占总面积的百分比
        area_percentage_distribution = {}
        if total_area > 0:
            for key in area_sum_distribution:
                area_percentage_distribution[key] = (area_sum_distribution[key] / total_area) * 100
        else:
            for key in area_sum_distribution:
                area_percentage_distribution[key] = 0
        
        # 将像素面积转换为平方微米
        total_area_um2 = total_area * PIXEL_TO_UM2
        area_sum_distribution_um2 = {k: v * PIXEL_TO_UM2 for k, v in area_sum_distribution.items()}
        
        return {
            'filename': filename,
            'fat_count': fat_count,
            'total_area_pixels': total_area,
            'total_area_um2': total_area_um2,
            'avg_fat_area_pixels': total_area / fat_count if fat_count > 0 else 0,
            'avg_fat_area_um2': total_area_um2 / fat_count if fat_count > 0 else 0,
            'area_distribution': area_distribution,
            'area_sum_distribution_pixels': area_sum_distribution,
            'area_sum_distribution_um2': area_sum_distribution_um2,
            'area_percentage_distribution': area_percentage_distribution
        }
        
    except Exception as e:
        logging.error(f"处理图像 {img_path} 失败: {str(e)}")
        return None

def main():
    # 初始化路径
    image_dir = r'C:\Users\83953\Desktop\AN-NACH-20260317'
    output_dir = image_dir
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)
    
    setup_logging(output_dir)
    logging.info("开始脂肪细胞分析（面积单位：平方微米）")
    logging.info(f"像素到平方微米换算系数: {PIXEL_TO_UM2} um²/pixel")
    
    # 处理所有BMP图像
    results = []
    for filename in os.listdir(image_dir):
        if filename.upper().endswith('.BMP'):
            file_path = os.path.join(image_dir, filename)
            result = process_image(file_path, output_dir)
            if result:
                results.append(result)
                logging.info(
                    f"处理完成: {filename} - "
                    f"脂肪粒数: {result['fat_count']} - "
                    f"总面积: {result['total_area_um2']:.2f} um² - "
                    f"平均面积: {result['avg_fat_area_um2']:.2f} um²\n"
                    f"面积分布(数量): 0-1um²: {result['area_distribution']['0-1um²']} | "
                    f"1-5um²: {result['area_distribution']['1-5um²']} | "
                    f"5-15um²: {result['area_distribution']['5-15um²']} | "
                    f"15-50um²: {result['area_distribution']['15-50um²']} | "
                    f"50-200um²: {result['area_distribution']['50-200um²']} | "
                    f"200-500um²: {result['area_distribution']['200-500um²']} | "
                    f"500+um²: {result['area_distribution']['500+um²']}\n"
                    f"面积百分比: 0-1um²: {result['area_percentage_distribution']['0-1um²']:.2f}% | "
                    f"1-5um²: {result['area_percentage_distribution']['1-5um²']:.2f}% | "
                    f"5-15um²: {result['area_percentage_distribution']['5-15um²']:.2f}% | "
                    f"15-50um²: {result['area_percentage_distribution']['15-50um²']:.2f}% | "
                    f"50-200um²: {result['area_percentage_distribution']['50-200um²']:.2f}% | "
                    f"200-500um²: {result['area_percentage_distribution']['200-500um²']:.2f}% | "
                    f"500+um²: {result['area_percentage_distribution']['500+um²']:.2f}%"
                )
    
    # 保存CSV结果
    if results:
        # 将area_distribution字典展开为单独列
        expanded_results = []
        for result in results:
            new_result = {
                'filename': result['filename'],
                'fat_count': result['fat_count'],
                'total_area_pixels': result['total_area_pixels'],
                'total_area_um2': result['total_area_um2'],
                'avg_fat_area_pixels': result['avg_fat_area_pixels'],
                'avg_fat_area_um2': result['avg_fat_area_um2'],
                'count_0-1um²': result['area_distribution']['0-1um²'],
                'count_1-5um²': result['area_distribution']['1-5um²'],
                'count_5-15um²': result['area_distribution']['5-15um²'],
                'count_15-50um²': result['area_distribution']['15-50um²'],
                'count_50-200um²': result['area_distribution']['50-200um²'],
                'count_200-500um²': result['area_distribution']['200-500um²'],
                'count_500+um²': result['area_distribution']['500+um²'],
                'area_sum_0-1um²': result['area_sum_distribution_um2']['0-1um²'],
                'area_sum_1-5um²': result['area_sum_distribution_um2']['1-5um²'],
                'area_sum_5-15um²': result['area_sum_distribution_um2']['5-15um²'],
                'area_sum_15-50um²': result['area_sum_distribution_um2']['15-50um²'],
                'area_sum_50-200um²': result['area_sum_distribution_um2']['50-200um²'],
                'area_sum_200-500um²': result['area_sum_distribution_um2']['200-500um²'],
                'area_sum_500+um²': result['area_sum_distribution_um2']['500+um²'],
                'percentage_0-1um²': result['area_percentage_distribution']['0-1um²'],
                'percentage_1-5um²': result['area_percentage_distribution']['1-5um²'],
                'percentage_5-15um²': result['area_percentage_distribution']['5-15um²'],
                'percentage_15-50um²': result['area_percentage_distribution']['15-50um²'],
                'percentage_50-200um²': result['area_percentage_distribution']['50-200um²'],
                'percentage_200-500um²': result['area_percentage_distribution']['200-500um²'],
                'percentage_500+um²': result['area_percentage_distribution']['500+um²']
            }
            expanded_results.append(new_result)
        
        df = pd.DataFrame(expanded_results)
        csv_path = os.path.join(output_dir, 'fat_analysis_results_um.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"分析结果已保存至 {csv_path}")
    
        # 计算统计信息
        stats = {
            '总图片数': len(results),
            '总脂肪粒数': df['fat_count'].sum(),
            '脂肪粒平均数量': df['fat_count'].mean(),
            '总脂肪面积(um²)': df['total_area_um2'].sum(),
            '平均脂肪面积(um²)': df['total_area_um2'].sum() / df['fat_count'].sum() if df['fat_count'].sum() > 0 else 0,
            '面积分布(0-1um²)': df['count_0-1um²'].sum(),
            '面积分布(1-5um²)': df['count_1-5um²'].sum(),
            '面积分布(5-15um²)': df['count_5-15um²'].sum(),
            '面积分布(15-50um²)': df['count_15-50um²'].sum(),
            '面积分布(50-200um²)': df['count_50-200um²'].sum(),
            '面积分布(200-500um²)': df['count_200-500um²'].sum(),
            '面积分布(500+um²)': df['count_500+um²'].sum()
        }
        logging.info("汇总统计:\n" + "\n".join(f"{k}: {v:.2f}" for k,v in stats.items()))
    else:
        logging.warning("未找到有效的BMP图像或所有处理均失败")

if __name__ == '__main__':
    main()
