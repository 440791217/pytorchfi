from ultralytics import YOLO
from PIL import Image
import cv2
import os
import argparse

def predict_with_model(model_path, image_path, conf=0.5, save_dir=None):
    """使用训练好的模型进行预测"""
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            im.save(save_path)
            print(f"预测结果已保存至: {save_path}")
        else:
            im.show()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用YOLOv8模型进行红绿灯检测')
    parser.add_argument('--model', type=str, default='runs/detect/yolov8n_traffic_lights/weights/best.pt', help='模型路径')
    parser.add_argument('--image', type=str, required=True, help='图像路径或目录')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--save-dir', type=str, help='保存预测结果的目录')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.image):
        # 处理目录中的所有图像
        for filename in os.listdir(args.image):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(args.image, filename)
                print(f"正在处理: {img_path}")
                predict_with_model(args.model, img_path, args.conf, args.save_dir)
    else:
        # 处理单张图像
        predict_with_model(args.model, args.image, args.conf, args.save_dir)