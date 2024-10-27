# -*- coding: utf-8 -*-
# @Time    : 2024/7/12 下午7:07
# @Author  : Dreamstar
# @File    : 1.Training a LeNet with MC Dropout.py
# @Desc    : 
# @Link    : https://torch-uncertainty.github.io/auto_tutorials/tutorial_mc_dropout.html#sphx-glr-auto-tutorials-tutorial-mc-dropout-py


from pathlib import Path

from lightning.pytorch import Trainer
# https://lightning.ai/docs/pytorch/stable/
from torch import nn

from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.models import mc_dropout
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import ClassificationRoutine


trainer = Trainer(accelerator="cpu", max_epochs=2, enable_progress_bar=False)

# datamodule
root = Path("data")
datamodule = MNISTDataModule(root=root, batch_size=128)


model = lenet(
    in_channels=datamodule.num_channels,
    num_classes=datamodule.num_classes,
    dropout_rate=0.5,
)

mc_model = mc_dropout(model, num_estimators=16, last_layer=False)


routine = ClassificationRoutine(
    num_classes=datamodule.num_classes,
    model=mc_model,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_cifar10_resnet18(mc_model),
    is_ensemble=True,
)


trainer.fit(model=routine, datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


dataiter = iter(datamodule.val_dataloader())
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images[:6, ...], padding=0))
print("Ground truth labels: ", " ".join(f"{labels[j]}" for j in range(6)))

routine.eval()
logits = routine(images).reshape(16, 128, 10)

probs = torch.nn.functional.softmax(logits, dim=-1)


for j in range(6):
    values, predicted = torch.max(probs[:, j], 1)
    print(
        f"Predicted digits for the image {j+1}: ",
        " ".join([str(image_id.item()) for image_id in predicted]),
    )


