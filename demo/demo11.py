#故障注入
import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
from pytorchfi.core import FaultInjection as fault_injection


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
torch.random.manual_seed(5)

# Model prep
# 使用 weights 参数加载预训练的 AlexNet 模型
model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
model = model.to(device)  # 将模型移动到指定设备
model.eval()

# generic image preperation
batch_size = 1
h = 224
w = 224
c = 3

image = torch.rand((batch_size, c, h, w))
image = image.to(device)  # 将输入图像移动到指定设备

output = model(image)

golden_label = list(torch.argmax(output, dim=1))[0].item()
print("Error-free label:", golden_label)

pfi_model = fault_injection(model,
                            batch_size,
                            input_shape=[c,h,w],
                            layer_types=[torch.nn.Conv2d],
                            use_cuda=True,
                            )
#
print(pfi_model.print_pytorchfi_layer_summary())
#
b, layer, C, H, W, err_val = [0], [2], [4], [2], [4], [10000]

inj = pfi_model.declare_neuron_fault_injection(batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val)

inj_output = inj(image)
inj_label = list(torch.argmax(inj_output, dim=1))[0].item()
print("[Single Error] PytorchFI label:", inj_label)