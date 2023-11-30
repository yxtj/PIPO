import torch
import model.resnet
from system.util import compute_shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.resnet.build(50).to(device)
inshape =(3, 224, 224)
# inp = torch.randn(1, 3, 224, 224).to(device)

shapes = compute_shape(model, inshape)
print(shapes)

for i in range(len(model)):
    print(i, model[i], shapes[i], shapes[i+1])


