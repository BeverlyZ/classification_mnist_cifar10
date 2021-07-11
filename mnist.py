import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.__version__

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        # nn.MaxPool2d
        # def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
        #             padding: _size_any_t = 0, dilation: _size_any_t = 1,
        #             return_indices: bool = False, ceil_mode: bool = False) -> None:
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


BATCH_SIZE=512
EPOCHS=1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', transform=transform, train=True, download=True,), BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',transform=transform, train=False), BATCH_SIZE, shuffle=True)

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

# 训练
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 更新权重和偏置
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({(100. * batch_idx / len(train_loader)):.0f}%)]\tLoss: {loss.item():.6f}")
            # print(f"Train Epoch: %s" % (epoch))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(100. * correct / len(test_loader.dataset)):.0f}%)\n")
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

# 开始训练
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

# 保存模型
torch.save(model, './model/model_cnn_1.pkl')