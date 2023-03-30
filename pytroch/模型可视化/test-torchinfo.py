# 官方的例子
from torchinfo import summary
import torchvision

model = torchvision.models.resnet152()
summary(model, (1, 3, 224, 224), depth=3)