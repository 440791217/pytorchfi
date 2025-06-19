from ultralytics import YOLO
import os
import shutil
from PIL import Image
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split

# 训练函数
def train_yolov8_model(data_config, model_size='nano', epochs=100, imgsz=640):
    """使用YOLOv8训练红绿灯检测模型"""
    # 修正模型名称映射
    model_mapping = {
        'nano': 'n',
        'small': 's',
        'medium': 'm',
        'large': 'l',
        'extra': 'x'
    }
    
    # 获取正确的模型后缀
    model_suffix = model_mapping.get(model_size.lower(), 'n')
    model_name = f"yolov8{model_suffix}.pt"
    
    # 加载预训练模型
    model = YOLO(model_name)
    
    # 训练模型
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name=f"yolov8{model_suffix}_traffic_lights"
    )
    
    # 验证模型
    model.val()
    
    return model

# 预测函数
def predict_with_model(model, image_path, conf=0.5):
    """使用训练好的模型进行预测"""
    # 进行预测
    results = model(image_path, conf=conf)
    
    # 显示结果
    for r in results:
        im_array = r.plot()  # 绘制带有预测框的图像
        im = Image.fromarray(im_array[..., ::-1])  # 转换为PIL图像
        im.show()  # 显示图像
    
    return results

# 主函数
def main():
    # 数据集配置文件路径
    data_config = "D:/jsut/datasets/交通信号灯/data.yaml"
    
    # 训练模型
    model = train_yolov8_model(data_config, model_size='nano', epochs=50)
    
    # 保存模型
    model.export(format='onnx')  # 导出为ONNX格式，便于部署
    
    # 示例预测
    # test_image = "path/to/test/image.jpg"
    # predict_with_model(model, test_image, conf=0.4)

if __name__ == "__main__":
    main()