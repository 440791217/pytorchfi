import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
from pytorchfi.core import FaultInjection as fault_injection

# List available models
all_models = models.list_models()
print(all_models)
classification_models = models.list_models(module=models)
print(classification_models)