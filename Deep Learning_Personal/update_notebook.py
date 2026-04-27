import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell

nb_path = 'C:/Users/Debaditya/Documents/Learning_ML/Deep Learning/Convolutional_Networks.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

cells = []

# Helper Functions
md_helper = """## 8. Helper Functions for Training and Evaluation
To compare multiple architectures compactly, we define a generic function to train and evaluate a model for a few epochs.
"""
code_helper = """import time
import torch
import torch.nn as nn
import torch.optim as optim

def train_and_evaluate(model, num_epochs=5, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    end_time = time.time()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Test Accuracy: {acc:.2f}% | Parameters: {num_params:,} | Time: {end_time - start_time:.2f}s")
    return acc, num_params, end_time - start_time
"""
cells.extend([new_markdown_cell(md_helper), new_code_cell(code_helper)])

# AlexNet
md_alexnet = """## 9. AlexNet
**Architecture:** Proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. It was a massive breakthrough that popularized Convolutional Neural Networks and GPU-accelerated training.

### Key Innovations
* **ReLU Non-linearity:** Replaced `tanh` and `sigmoid` activation functions, dramatically speeding up training and helping mitigate the vanishing gradient problem.
* **Dropout:** Introduced Dropout layers to reduce overfitting in the massive fully connected layers.
* **Overlapping Pooling:** Used pooling windows larger than the stride, which was shown to slightly reduce top-1 and top-5 error rates.

### Conceptual Architecture (Adapted for CIFAR-10)
```text
Input Image (3x32x32)
│
├─► Conv2d(64, 3x3) + ReLU ──► MaxPool(2x2)
├─► Conv2d(192, 3x3) + ReLU ──► MaxPool(2x2)
├─► Conv2d(384, 3x3) + ReLU
├─► Conv2d(256, 3x3) + ReLU
├─► Conv2d(256, 3x3) + ReLU ──► MaxPool(2x2)
│
├─► Flatten
├─► Linear(1024) + ReLU + Dropout
├─► Linear(1024) + ReLU + Dropout
└─► Linear(10) (Output)
```
*Adaptation for CIFAR-10:* Standard AlexNet uses 11x11 convs with stride 4, which aggressively reduces 224x224 images. For 32x32 images, we use smaller kernels (3x3) and fewer channels/pooling layers to prevent the spatial dimensions from collapsing prematurely.
"""
code_alexnet = """class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

print("--- Training AlexNet ---")
alexnet_model = AlexNetCIFAR()
alex_acc, alex_params, alex_time = train_and_evaluate(alexnet_model, num_epochs=5)
"""
cells.extend([new_markdown_cell(md_alexnet), new_code_cell(code_alexnet)])

# VGG-19
md_vgg = """## 10. VGG-19
**Architecture:** Proposed by the Visual Geometry Group (VGG) at Oxford (2014). VGG architectures are characterized by their simplicity, depth, and uniformity.

### Key Innovations
* **Uniform 3x3 Convolutions:** VGG demonstrated that depth is crucial. By stacking multiple 3x3 convolutions, VGG mimics the receptive field of larger convolutions (e.g., two 3x3s = one 5x5, three 3x3s = one 7x7).
* **Fewer Parameters, More Non-linearity:** A 7x7 convolution with $C$ channels has $49C^2$ parameters. Three 3x3 convolutions have $3 \\times 9C^2 = 27C^2$ parameters. This design reduces the number of parameters while inserting three ReLU non-linearities instead of one, making the decision function more discriminative.

### Conceptual Architecture (VGG-19)
```text
Input (3x32x32)
│
├─► [Conv3x3(64) + ReLU] x 2  ──► MaxPool(2x2)
├─► [Conv3x3(128) + ReLU] x 2 ──► MaxPool(2x2)
├─► [Conv3x3(256) + ReLU] x 4 ──► MaxPool(2x2)
├─► [Conv3x3(512) + ReLU] x 4 ──► MaxPool(2x2)
├─► [Conv3x3(512) + ReLU] x 4 ──► MaxPool(2x2)
│
├─► Flatten
└─► Linear(10) (Output)
```
*Adaptation for CIFAR-10:* Standard VGG uses three massive Fully Connected layers at the end. Because 5 pooling layers reduce 32x32 spatial dimensions down to 1x1, we map directly to a smaller linear layer.
"""
code_vgg = """cfg_vgg19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG19CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19CIFAR, self).__init__()
        layers = []
        in_channels = 3
        for x in cfg_vgg19:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

print("--- Training VGG-19 ---")
vgg_model = VGG19CIFAR()
vgg_acc, vgg_params, vgg_time = train_and_evaluate(vgg_model, num_epochs=5)
"""
cells.extend([new_markdown_cell(md_vgg), new_code_cell(code_vgg)])

# GoogleNet
md_googlenet = """## 11. GoogleNet (Inception v1)
**Architecture:** Proposed by Google (Szegedy et al., 2014), it introduced the highly efficient "Inception Module," pushing networks to be deeper and wider without blowing up parameter counts.

### Key Innovations
* **Inception Module:** Instead of manually choosing whether a layer should be a 1x1, 3x3, or 5x5 convolution, or a max-pooling layer, the Inception module does *all of them* in parallel and concatenates the feature maps.
* **1x1 Convolutions (Bottlenecks):** Used extensively to compute channel reductions before the expensive 3x3 and 5x5 convolutions, drastically reducing computational cost.
* **Global Average Pooling:** Replaced the dense, massive fully connected layers at the end of the network, contributing heavily to parameter reduction (GoogleNet has ~4M parameters vs AlexNet's ~60M).

### Conceptual Architecture (Inception Module)
```text
                     ┌──► 1x1 Conv ───────────────┐
                     │                            │
                     ├──► 1x1 Conv ──► 3x3 Conv ──┼─► Concat
Input ──► Previous ──┤                            │   Output
Layer                ├──► 1x1 Conv ──► 5x5 Conv ──┤
                     │                            │
                     └──► 3x3 MaxPool ─► 1x1 Conv ┘
```
*Adaptation for CIFAR-10:* We use a lighter version of the Inception network that bypasses the aggressive initial 7x7 strided pooling, which is better suited for 32x32 inputs.
"""
code_googlenet = """class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class GoogleNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNetCIFAR, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.a3 = InceptionModule(192,  64,  96, 128, 16, 32, 32)
        self.b3 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = InceptionModule(480, 192,  96, 208, 16,  48,  64)
        self.b4 = InceptionModule(512, 160, 112, 224, 24,  64,  64)
        self.c4 = InceptionModule(512, 128, 128, 256, 24,  64,  64)
        self.d4 = InceptionModule(512, 112, 144, 288, 32,  64,  64)
        self.e4 = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.a5 = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

print("--- Training GoogleNet ---")
googlenet_model = GoogleNetCIFAR()
googlenet_acc, googlenet_params, googlenet_time = train_and_evaluate(googlenet_model, num_epochs=5)
"""
cells.extend([new_markdown_cell(md_googlenet), new_code_cell(code_googlenet)])

# ResNet
md_resnet = """## 12. ResNet (Residual Networks)
**Architecture:** Proposed by Kaiming He et al. (2015). It revolutionized deep learning by effectively solving the "degradation problem," allowing networks to scale up to hundreds of layers.

### Key Innovations
* **Skip Connections (Residual Blocks):** Instead of trying to learn an unreferenced mapping $H(x)$, ResNet aims to learn a residual function $F(x) = H(x) - x$. The original input $x$ is added back to the learned features via a "skip connection".
* **Solving the Degradation Problem:** Before ResNet, simply adding more layers to a deep network often caused the training error to *increase* due to optimization issues (vanishing gradients). Skip connections allow gradients to bypass layers and flow backwards directly, solving this.
* **Batch Normalization:** ResNet relies heavily on Batch Normalization after convolutions to stabilize training.

### Conceptual Architecture (Residual Block)
```text
        x (Input)
        │
        ├───► Conv(3x3) ──► BN ──► ReLU ──► Conv(3x3) ──► BN ──┐
        │                                                      │
        └─────────────────────── Identity ─────────────────────┼──► + ──► ReLU ──► Output
                                                                  (F(x) + x)
```
*Adaptation for CIFAR-10:* We implement the classic ResNet-18 adapted for 32x32 images. We modify the standard ResNet by replacing the initial 7x7 conv and 3x3 maxpool with a single 3x3 conv, preserving spatial resolution for smaller images.
"""
code_resnet = """class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

def ResNet18CIFAR():
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2])

print("--- Training ResNet-18 ---")
resnet_model = ResNet18CIFAR()
resnet_acc, resnet_params, resnet_time = train_and_evaluate(resnet_model, num_epochs=5)
"""
cells.extend([new_markdown_cell(md_resnet), new_code_cell(code_resnet)])

# Comparison
md_comparison = """## 13. Comparison of Results
Finally, we can compare the 4 added CNN architectures below in terms of Test Accuracy, Number of Parameters, and Training Time (for 5 epochs).
"""
code_comparison = """import matplotlib.pyplot as plt
import numpy as np

# Assuming we recorded stats:
models_names = ['AlexNet', 'VGG-19', 'GoogleNet', 'ResNet-18']
acc_list = [alex_acc, vgg_acc, googlenet_acc, resnet_acc]
params_list = [alex_params, vgg_params, googlenet_params, resnet_params]
time_list = [alex_time, vgg_time, googlenet_time, resnet_time]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('CNN Architecture')
ax1.set_ylabel('Test Accuracy (%)', color=color)
bars = ax1.bar(models_names, acc_list, color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Parameters (Millions)', color=color)  
ax2.plot(models_names, [p / 1e6 for p in params_list], color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Comparison of CNN Architectures on CIFAR-10 (5 Epochs)')

# Add annotations to bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', color='black', fontweight='bold')

plt.tight_layout()
plt.show()

print("--- Summary Report ---")
for i, name in enumerate(models_names):
    print(f"{name:12s} | Acc: {acc_list[i]:.2f}% | Params: {params_list[i]:>10,d} | Time: {time_list[i]:.2f}s")
"""
cells.extend([new_markdown_cell(md_comparison), new_code_cell(code_comparison)])

nb.cells.extend(cells)
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Updated Notebook effectively.")