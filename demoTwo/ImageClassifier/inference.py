import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

# --- same ResNet18 definition as before ----------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
# -------------------------------------------------------

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reload trained model
model = ResNet18().to(device)
model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location=device))
model.eval()

# dataset + transform (test set only)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='cifar10', train=False,
                                       download=True, transform=transform_test)

# class labels
classes = testset.classes

# random viewer logic
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # leave space for button
img_ax = ax.imshow(torchvision.transforms.ToPILImage()(testset[0][0]))
title = ax.set_title("")

def show_random(_=None):
    idx = random.randint(0, len(testset)-1)
    image, label = testset[idx]
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))
        _, pred = torch.max(out, 1)
    # update plot
    img_ax.set_data(torchvision.transforms.ToPILImage()(image))
    title.set_text(f"Pred: {classes[pred.item()]} | True: {classes[label]}")
    fig.canvas.draw_idle()

# add Next button
axnext = plt.axes([0.4, 0.05, 0.2, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(show_random)

# show first random image
show_random()

plt.show()
