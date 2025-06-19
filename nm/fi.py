import torch
import torchvision.models as models

from pytorchfi.core import FaultInjection
from pytorchfi.neuron_error_models import single_bit_flip_func


torch.random.manual_seed(5)

# Model prep
model = models.alexnet(pretrained=True)
model.eval()

# generic image preperation
batch_size = 1
h = 224
w = 224
c = 3

image = torch.rand((batch_size, c, h, w))

output = model(image)

golden_label = list(torch.argmax(output, dim=1))[0].item()
print("Error-free label:", golden_label)


# pfi_model = FaultInjection(model, 
#                      batch_size,
#                      input_shape=[c,h,w],
#                      layer_types=[torch.nn.Conv2d],
#                      use_cuda=False,
#                      )

pfi_model = single_bit_flip_func(model, 
                     batch_size,
                     input_shape=[c,h,w],
                     layer_types=[torch.nn.Conv2d],
                     use_cuda=False,
                     )

print(pfi_model.print_pytorchfi_layer_summary())

b, layer, C, H, W, err_val = [0], [3], [4], [2], [4], [10000]

# inj = pfi_model.declare_neuron_fault_injection(batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val)
inj=pfi_model.declare_neuron_fault_injection(function=pfi_model.single_bit_flip_signed_across_batch)

inj_output = inj(image)
inj_label = list(torch.argmax(inj_output, dim=1))[0].item()
print("[Single Error] PytorchFI label:", inj_label)