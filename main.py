import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from bs_resnet import my_ResNet18, train, test, test_repeat
from memristor_optimizer import Memristor, ltp_ltd_plot, OptimizerAdamMem

import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print('Device:', device)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if not os.path.isdir('train_results_resnet'):
    os.mkdir('train_results_resnet')

print('Data prepared.')

# Model
filename='resnet18_Sigmoid_bs'
max_epoch=200

print('==> Building model..')
net = my_ResNet18(act_type='Sigmoid_BS') #ReLU_BS
net = net.to(device)

net2 = my_ResNet18(act_type='Sigmoid') #ReLU_T
net2 = net2.to(device)


# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

assert os.path.isdir('train_results_resnet'), 'Error: no checkpoint directory found!'
if os.path.isfile(f'./train_results_resnet/{filename}.pth'):
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(f'./train_results_resnet/{filename}.pth',weights_only=True,map_location=device)
    net.load_state_dict(checkpoint['net'])
    acc_train_epoch = checkpoint['acc_train_epoch']
    acc_test_hp_epoch = checkpoint['acc_test_hp_epoch']
    acc_test_bs_epoch = checkpoint['acc_test_bs_epoch']
    loss_train_epoch = checkpoint['loss_train_epoch']
    weight_updates_epoch = checkpoint['weight_updates_epoch']
    start_epoch = len(acc_train_epoch)
else:
    loss_train_epoch=[]
    acc_train_epoch=[]
    acc_test_hp_epoch=[]
    acc_test_bs_epoch=[]
    loss_train_epoch = []
    weight_updates_epoch = []
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

memristor_parameters = {
        'mode': 'lin',
        'Wmin': -1.0, 'Wmax': 1.0,
        'Gmin': 2.0e-6, 'Gmax': 3.2e-5,
        'alpha_p': 0.0, 'alpha_d': 0.0,
        'Np':8, 'Nd': 8,
        'gamma_p': 0.0, 'gamma_d': 0.0,
        'gamma_gmin': 0.0, 'gamma_gmax': 0.0
    }

ltp_ltd_plot(memristor_parameters,filename = f'./train_results_resnet/{filename}')
total_weights = 0
for p in net.parameters():
    p.memristor = Memristor(p, **memristor_parameters)
    total_weights += p.data.numel()

optimizer = OptimizerAdamMem(net.parameters(), betas=(0.9, 0.999), eps=1e-8, weight_decay=0, threshold=10.0)

print('==> Starting training..')
for epoch in range(start_epoch, max_epoch):
    if epoch>=100:
        optimizer.threshold = 100.0
    print(f'Epoch: {epoch+1}/{max_epoch}')
    acc_train,loss,weight_updates = train(trainloader, net, optimizer, device)
    print(f"Weight updates: {weight_updates}, Total weights: {total_weights}, Update ratio: {weight_updates/total_weights:.4f}")

    acc_test_bs = test(testloader, net, device)

    net2.load_state_dict(net.state_dict())
    net2.eval()  
    acc_test_hp = test(testloader, net2, device)

    print('Saving..')
    loss_train_epoch.append(loss)
    acc_train_epoch.append(acc_train)
    acc_test_hp_epoch.append(acc_test_hp)
    acc_test_bs_epoch.append(acc_test_bs)
    weight_updates_epoch.append(weight_updates)
    state = {
        'net': net.state_dict(),
        'acc_train_epoch': acc_train_epoch,
        'acc_test_hp_epoch': acc_test_hp_epoch,
        'acc_test_bs_epoch': acc_test_bs_epoch,
        'loss_train_epoch': loss_train_epoch,
        'weight_updates_epoch': weight_updates_epoch,
        'epoch': epoch,
    }
    torch.save(state, f'./train_results_resnet/{filename}.pth')


    fig = plt.figure()
    plt.plot(range(1,len(acc_train_epoch)+1), np.array(acc_train_epoch)*100, marker='o', label=f'TrainSet ({net.act_type})')
    plt.plot(range(1,len(acc_test_hp_epoch)+1), np.array(acc_test_hp_epoch)*100, marker='o', label=f'TestSet ({net2.act_type})')
    plt.plot(range(1,len(acc_test_bs_epoch)+1), np.array(acc_test_bs_epoch)*100, marker='o', label=f'TestSet ({net.act_type})')
    plt.title(f'Accuracy vs Epochs ({filename})')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.savefig(f'./train_results_resnet/{filename}_accuracy.svg')
    plt.close()

    fig = plt.figure()
    plt.plot(range(1,len(loss_train_epoch)+1), loss_train_epoch, marker='o', label='Training Loss')
    plt.title(f'Training Loss vs Epochs ({filename})')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig(f'./train_results_resnet/{filename}_loss.svg')
    plt.close() 

    fig = plt.figure()
    plt.plot(range(1,len(weight_updates_epoch)+1), weight_updates_epoch, marker='o', label='Weight Updates')
    plt.title(f'Weight Updates vs Epochs ({filename})')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Weight Updates')
    plt.legend()
    plt.savefig(f'./train_results_resnet/{filename}_weight_updates.svg')
    plt.close()

print('==> Finished Training.')

for i, acc in enumerate(acc_train_epoch):
    print(f'Epoch {i+1}: trainset (Sigmoid_BS) {acc*100:.3f}%, testset (Sigmoid_BS) {acc_test_hp_epoch[i]*100:.3f}%, testset (Sigmoid_BS) {acc_test_bs_epoch[i]*100:.3f}%')


num_repeats = 50
checkpoint = torch.load(f'./train_results_resnet/{filename}.pth',weights_only=True,map_location=device)
if 'acc_test_bs_repeats' in checkpoint and len(checkpoint['acc_test_bs_repeats'])>=num_repeats:
    acc_test_bs_repeats = checkpoint['acc_test_bs_repeats']
else:
    net.load_state_dict(checkpoint['net'])
    net.eval()  
    acc_test_bs_repeats = test_repeat(testloader, net, num_repeat=num_repeats, device=device)
    checkpoint['acc_test_bs_repeats'] = acc_test_bs_repeats
    torch.save(checkpoint, f'./train_results_resnet/{filename}.pth')

for i, acc in enumerate(acc_test_bs_repeats):
    print(f'Testset ({net.act_type}) repeat {i+1}: {acc*100:.3f}%')

fig = plt.figure()
plt.plot(range(1,len(acc_test_bs_repeats)+1), np.ones_like(acc_test_bs_repeats)*acc_test_hp_epoch[-1]*100, label=f'TestSet ({net2.act_type})')
plt.plot(range(1,len(acc_test_bs_repeats)+1), np.array(acc_test_bs_repeats)*100, label=f'TestSet ({net.act_type}) - Repeats', marker='o')
plt.title(f'Accuracy vs Repeats ({filename})')
plt.grid()
plt.xlabel('Repeats')
plt.ylabel('Accuracy [%]')
plt.legend()
plt.savefig(f'./train_results_resnet/{filename}_accuracy_bs_repeats.svg')
plt.close()

