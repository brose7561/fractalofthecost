import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# training params
num_epochs = 35
learning_rate = 0.1

# data pipeline with weak augmentation for regularisation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
])


# test set pipeline with only normalisation (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])




# load CIFAR-10 training set with augmentation
trainset = torchvision.datasets.CIFAR10(
    root='cifar10', train=True, download=True, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True  # shuffle for randomness each epoch
)

# load CIFAR-10 test set with only normalisation
testset = torchvision.datasets.CIFAR10(
    root='cifar10', train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False  # keep order fixed for evaluation
)


class BasicBlock(nn.Module):
    expansion = 1  # resnet18/34 block does not expand channel count

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # first conv -> bn -> relu
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # second conv -> bn (relu applied later after residual add)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # shortcut to match dimensions when stride or channel count changes
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
        out += self.shortcut(x)  # add residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # first conv and batchnorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # build the 4 layers of resnet
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # final linear classifier
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # each layer can downsample once at start, then stride=1
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

# build resnet18
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# build and move model to device
model = ResNet18()
model = model.to(device)

# show gpu info if available
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# model info
print("Model No. of Parameters:", sum(param.nelement() for param in model.parameters()))
print(model)


# loss function
criterion = nn.CrossEntropyLoss()

# optimiser with momentum + weight decay for regularisation
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=0.9,
                            weight_decay=5e-4)

# learning rate schedule setup
total_step = len(train_loader)

# cyclic warmup schedule
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.005,
    max_lr=learning_rate,
    step_size_up=15,
    step_size_down=15,
    mode="triangular",
    verbose=False
)

# decay schedule near the end
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.005 / learning_rate,
    end_factor=0.005 / 5,
    verbose=False
)

# combine into sequential schedule
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[sched_linear_1, sched_linear_3],
    milestones=[30]
)



# train loop
model.train()
print("> Training")
start = time.time()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # move batch to gpu/cpu
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward + optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log every 100 steps
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{i+1}/{total_step}] "
                  f"Loss: {loss.item():.5f}")

    scheduler.step()

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " +
      str(elapsed/60) + " mins in total")




# test loop
print("> Testing")
start = time.time()
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # move batch to gpu/cpu
        images = images.to(device)
        labels = labels.to(device)

        # forward pass only
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # update counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " +
      str(elapsed/60) + " mins in total")
