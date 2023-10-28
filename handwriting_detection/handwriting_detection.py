'''
这个文件只用于train数据集，训练完之后，保存模型，train函数就可以被注释掉了

MNIST数据集下载到了同一目录下的data文件夹下的MNIST文件夹里
同一目录下的TestDigitImgs文件夹放着待自被测试的图片，其中第一张图片已经被自己编辑成28×28像素的大小，符合本项目模型的输入大小尺寸
'''
import torch
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from model import *
import cv2

# from test2 import imgNamepath

# from QFileDialogTest import *
# 在需要生成随机数的程序中，确保每次运行程序所生成的随机数都是固定的，使得实验结果一致
torch.manual_seed(1)
batch_size_train = 64
batch_size_valid = 64
batch_size_test = 1000

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    indices = range(len(testset))
    
    # 测试集中再取出一半作为验证集
    indices_valid = indices[:5000]
    sampler_valid = torch.utils.data.sampler.SubsetRandomSampler(indices_valid)
    validloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_valid, sampler=sampler_valid)
    
    indices_test = indices[5000:]
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, sampler=sampler_test)
    
    return trainloader, validloader, testloader



net = CNNNet()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
net.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.01)
trainloader, validloader, testloader = get_data()

# 训练损失
train_losses = []
# 训练进度
train_counter = []
# 验证损失
valid_losses = []
# 验证进度
valid_counter = []

def train(model, optimizer, loss_fn, train_loader, valid_loader, epochs=10, device='cpu'):
    for epoch in range(1, epochs+1):
        model.train()
        for train_idx, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 1.获取模型输出
            outputs = model(inputs)
            # 2.梯度清零
            optimizer.zero_grad()
            # 3.计算损失
            loss = loss_fn(outputs, labels)
            # 4.反向传播
            loss.backward()
            # 5.参数优化
            optimizer.step()
            if train_idx % 10 == 0:
                train_losses.append(loss.item())
                counter_index = train_idx * len(inputs) + (epoch-1) * len(train_loader.dataset)
                train_counter.append(counter_index)
                print('epoch: {}, [{}/{}({:.0f}%)], loss: {:.6f}'.format(
                    epoch, train_idx*len(inputs), len(train_loader.dataset), 100*(train_idx*len(inputs)+(epoch-1)*len(train_loader.dataset))/(len(train_loader.dataset)*(epochs)), loss.item()))
            
                # validation
                if train_idx % 300 == 0:
                    model.eval()
                    valid_loss = []
                    for valid_idx, (inputs, labels) in enumerate(valid_loader, 0):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        valid_loss.append(loss.item())
                    # 平均损失
                    valid_losses.append(np.average(valid_loss))
                    valid_counter.append(counter_index)
                    print('validation loss: {:.6f} counter_index: {}'.format((np.average(valid_loss)), counter_index))

    '''以下是修改内容'''
    torch.save(model, "./Models/CNNNet.pth".format(epoch))
    print("模型已保存")
    '''以上是修改部分'''

    print('training ended')



# train(net, optimizer, loss_fn, trainloader, validloader, epochs=2)

net = torch.load("./Models/CNNNet.pth", map_location=torch.device('cpu'))

'''以上是修改细节'''
# 平均测试损失
test_loss_avg = 0

def test(model, test_loader, loss_fn, device='cpu'):
    correct = 0
    total = 0
    test_loss = []
    with torch.no_grad():
        for train_idx, (inputs, labels) in enumerate(test_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss.append(loss.item())
            index, value = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += int((value==labels).sum())
        test_loss_avg = np.average(test_loss)
        print('Total: {}, Correct: {}, Accuracy: {:.2f}%, AverageLoss: {}'.format(total, correct, (correct/total*100), test_loss_avg))

test(net, testloader, loss_fn)

# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.plot(valid_counter, valid_losses, color='red')
# plt.scatter(train_counter[-1], test_loss_avg, color='green')
# plt.legend(['Train Loss', 'Valid Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('Training images number')
# plt.ylabel('Loss')
# plt.show()

import cv2
import os

rootdir = './TestDigitImgs'
list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
for i in range(0, len(list)):
    path = os.path.join(rootdir, list[i])
    img = cv2.imread(path, 0)
    height,width=img.shape
    dst=np.zeros((height,width),np.uint8)
    # 像素反转
    for i in range(height):
        for j in range(width):
            dst[i,j]=255-img[i,j]
    # 修改尺寸
    if height != 28 or width != 28:
        img = cv2.resize(dst, (28, 28))
    # 保存图片
    cv2.imwrite(path, dst)


from torch.autograd import Variable
import torch.nn.functional as F


def show():
    # img = cv2.imread('./TestDigitImgs/1600417425118.jpg', 0)
    # img = cv2.imread(imgNamePath, 0)
    img = cv2.imread('./TestDigitImgs/1600417457655.jpg', 0)
    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    img = img.to(device)
    output = net(Variable(img))
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()
    print(prob)
    pred = np.argmax(prob)
    print(pred.item())
    return pred.item()
# show()

# if __name__ == '__main__':
#     import sys
#
#     app = QtWidgets.QApplication(sys.argv)
#     formObj = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(formObj)
#     formObj.show()
#     sys.exit(app.exec_())
