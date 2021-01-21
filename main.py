import warnings
import random
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from src.datagenerator import MaskingDataset_train, MaskingDataset_validation, MaskingDataset_test
from src.net import Net
from src.preprocessing import Horizontal_Scaling_0_1, ToTensor, Horizontal_Scaling_m1_1
from src.gen_mask_traces import TraceGenerator
from src.config import Config

warnings.filterwarnings('ignore',category=FutureWarning)
config = Config()


#Seed the pipeline for reproductibility
seed = config.general.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#TODO: 1. preprocessing (Done)
#TODO: 2. validation sets (Done)
#TODO: 3. Neural Network

#The preprocessing in each sample.
compose = transforms.Compose([ToTensor()])
Horizontal_scale_0_1 = transforms.Compose([  ToTensor(), Horizontal_Scaling_0_1() ])
Horizontal_scale_m1_1 = transforms.Compose([  ToTensor(), Horizontal_Scaling_m1_1() ])

#LOAD trainset and feature preprocessing
trainset = -1
valset = -1
if config.dataloader.scaling == "None":
    trainset = MaskingDataset_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    valset = MaskingDataset_validation(config, X_validation, Y_validation)

elif config.dataloader.scaling == "horizontal_scale_0_1":
    trainset = MaskingDataset_train(config, transform=Horizontal_scale_0_1)
    X_validation, Y_validation = trainset.train_validation_split()
    valset = MaskingDataset_validation(config, X_validation, Y_validation)

elif config.dataloader.scaling == "horizontal_scale_m1_1":
    trainset = MaskingDataset_train(config, transform=Horizontal_scale_m1_1)
    X_validation, Y_validation = trainset.train_validation_split()
    valset = MaskingDataset_validation(config, X_validation, Y_validation)

elif config.dataloader.scaling == "feature_scaling_0_1":
    trainset = MaskingDataset_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    trainset.feature_min_max_scaling(0,1)
    valset = MaskingDataset_validation(config, X_validation, Y_validation, feature_scaler=trainset.get_feature_scaler())

elif config.dataloader.scaling == "feature_scaling_m1_1":
    trainset = MaskingDataset_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    trainset.feature_min_max_scaling(-1,1)
    valset = MaskingDataset_validation(config, X_validation, Y_validation,feature_scaler=trainset.get_feature_scaler())

elif config.dataloader.scaling == "feature_standardization":
    trainset = MaskingDataset_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    trainset.feature_standardization()
    valset = MaskingDataset_validation(config, X_validation, Y_validation, feature_scaler=trainset.get_feature_scaler())

valset.feature_scaling()

#trainset.to_categorical(num_classes=256)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.dataloader.batch_size,
                                          shuffle=config.dataloader.shuffle,
                                          num_workers=config.dataloader.num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=config.dataloader.batch_size,
                                          shuffle=config.dataloader.shuffle,
                                          num_workers=config.dataloader.num_workers)

dataloader = {"train": trainloader, "val": valloader}

testset = -1
if config.dataloader.scaling == "horizontal_scale_0_1":
    testset = MaskingDataset_test(config, transform=Horizontal_scale_0_1)
elif config.dataloader.scaling == "horizontal_scale_m1_1":
    testset = MaskingDataset_test(config, transform=Horizontal_scale_m1_1)
else:
    scaler = trainset.get_feature_scaler()
    testset= MaskingDataset_test(config, transform=compose, feature_scaler=scaler)
    testset.feature_scaling()

#testset.to_categorical(num_classes=256)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_dataloader.batch_size,
                                         shuffle=config.test_dataloader.shuffle,
                                        num_workers=config.test_dataloader.num_workers)
# print("Trainset:")
# print(len(trainset))
# for i in range(len(trainset)):
# #     print(trainset[i])
#      print(trainset[i]["trace"])

# print("Testset:")
# for i in range(len(testset)):
#     print("testset " + str(i))
#     print(testset[i]["trace"])

#
net = Net()
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data["trace"].float(), data["sensitive"].float()
        #print(inputs.shape)
        # zero the parameter gradients
        # optimizer.zero_grad()
        #
        # # forward + backward + optimize
        # outputs = net(inputs)
        # print(outputs)
        # loss = criterion(outputs, labels.squeeze(-1))
        #
        # loss.backward()
        # optimizer.step()
        #
        # # print statistics
        # running_loss += loss.item()
        # if i % 20 == 19:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 20))
        #     running_loss = 0.0
        #     correct = 0
        #     total = 0
        #     with torch.no_grad():
        #         for data in testloader:
        #             images, labels = data
        #             outputs = net(images)
        #             _, predicted = torch.max(outputs.data, 1)
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()
        #
        #     print('Accuracy of the network on the 10000 test images: %d %%' % (
        #             100 * correct / total))

print('Finished Training')





