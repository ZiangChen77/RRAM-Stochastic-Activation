import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from bs_activation import ReLU, ReLU_T, ReLU_BS, Sigmoid, Sigmoid_BS, CrossEntropy, CrossEntropy_BS

'''Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, activation, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.activation=activation
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, activation, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.activation=activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,act_type='ReLU'):
        super(ResNet, self).__init__()
        self.act_type=act_type
        self.in_planes = 64
        if act_type=='ReLU':
            self.activation=ReLU.apply
        elif act_type=='ReLU_T':
            self.activation=ReLU_T.apply
        elif act_type=='ReLU_BS':
            self.activation=ReLU_BS.apply
        elif act_type=='Sigmoid':
            self.activation=Sigmoid.apply
        elif act_type=='Sigmoid_BS':
            self.activation=Sigmoid_BS.apply
        if "BS" in self.act_type:
            self.cross_entropy = CrossEntropy_BS.apply
        else:
            self.cross_entropy = CrossEntropy.apply

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.activation,self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target_one_hot):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        logits, loss = self.cross_entropy(out,target_one_hot)
        return logits, loss


def my_ResNet18(act_type='ReLU'):
    return ResNet(BasicBlock, [2, 2, 2, 2], act_type=act_type)


def my_ResNet34(act_type='ReLU'):
    return ResNet(BasicBlock, [3, 4, 6, 3], act_type=act_type)


def my_ResNet50(act_type='ReLU'):
    return ResNet(Bottleneck, [3, 4, 6, 3], act_type=act_type)


def my_ResNet101(act_type='ReLU'):
    return ResNet(Bottleneck, [3, 4, 23, 3], act_type=act_type)


def my_ResNet152(act_type='ReLU'):
    return ResNet(Bottleneck, [3, 8, 36, 3], act_type=act_type)

def train(trainloader, net, optimizer, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_updates = 0
    pbar = tqdm(trainloader, desc=f"Train ({net.act_type})", leave=True)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        y_one_hot=nn.functional.one_hot(targets,num_classes=10)
        logits, loss = net(inputs, y_one_hot)

        loss.backward()
        updates = optimizer.step(batch_idx+1)

        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_updates += updates

        pbar.set_postfix({
            'Loss': '%.3f' % (train_loss/(batch_idx+1)),
            'Acc': '%.3f%%' % (100.*correct/total)
        })
    return correct/total, train_loss/(batch_idx+1), total_updates


def test(testloader, net, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(testloader, desc=f"Test ({net.act_type})", leave=True)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            y_one_hot=nn.functional.one_hot(targets,num_classes=10)
            outputs, _ = net(inputs, y_one_hot)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Acc': '%.3f%%' % (100.*correct/total)
            })
    return correct/total

def test_repeat(testloader, net, num_repeat=10, device='cpu'):
    net.eval()

    with torch.no_grad():
        
        y_one_hot_batch=[]
        correct_rep = []
        for rep in range(num_repeat):
            correct = 0
            total = 0
            pbar = tqdm(testloader, desc=f"Test ({net.act_type}), repeat:{rep+1}", leave=True)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                if rep==0:
                    y_one_hot_batch.append(torch.zeros(len(targets),10).to(device))
                inputs, targets = inputs.to(device), targets.to(device)
                y_one_hot=nn.functional.one_hot(targets,num_classes=10)
            
                outputs, _ = net(inputs, y_one_hot)

                y_one_hot_batch[batch_idx]+=nn.functional.one_hot(outputs.argmax(dim=1),num_classes=10)

                _, predicted = y_one_hot_batch[batch_idx].max(dim=1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.set_postfix({
                        'Acc': '%.3f%%' % (100.*correct/total)
                    })
            correct_rep.append(correct/total)
    return correct_rep