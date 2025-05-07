import torch
from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights

# 生成随机图像数据，假设图像通道为 3，大小为 224x224
# 这里的尺寸是根据常见的图像分类模型输入要求设定的
img = torch.rand(3, 224, 224)
# 设置 fbgemm 作为量化后端
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
# Step 1: Initialize model with the best available weights
weights = MobileNet_V2_QuantizedWeights.DEFAULT
model = mobilenet_v2(weights=weights, quantize=True)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score}%")
    