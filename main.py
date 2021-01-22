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
from torch.utils.tensorboard import SummaryWriter


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
# Load Testset.
# testset = -1
# if config.dataloader.scaling == "horizontal_scale_0_1":
#     testset = MaskingDataset_test(config, transform=Horizontal_scale_0_1)
# elif config.dataloader.scaling == "horizontal_scale_m1_1":
#     testset = MaskingDataset_test(config, transform=Horizontal_scale_m1_1)
# else:
#     scaler = trainset.get_feature_scaler()
#     testset= MaskingDataset_test(config, transform=compose, feature_scaler=scaler)
#     testset.feature_scaling()
#
#
# #testset.to_categorical(num_classes=256)
# testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_dataloader.batch_size,
#                                          shuffle=config.test_dataloader.shuffle,
#                                         num_workers=config.test_dataloader.num_workers)
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
writer = SummaryWriter("runs/first run")

net = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=float(config.train.lr))

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = float(config.train.lr), epochs = config.train.epochs, steps_per_epoch=len(trainloader))
#Plot in tensorboard the curves loss and accuracy for train and val
for epoch in range(config.train.epochs):  # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch+1, config.train.epochs))
    print('-' * 10)
    for phase in ["train", "val"]:
        if phase == "train":
            net.train()
        else:
            net.eval()

        running_loss = 0.0

        for i, data in enumerate(dataloader[phase], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["trace"].float(), data["maskorder"].float()
            #print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            ## TODO: Set up accuracy for both train set and var set.
            # correct = 0
            # total = 0
            # forward + backward + optimize
            with torch.set_grad_enabled(phase == "train"):
                outputs = net(inputs)
                labels = labels.view(int(config.dataloader.batch_size)).long() ##This is because NLLLoss only take in this form.
                outputs = torch.log(outputs) #### NLLloss + softmax + log = CrossEntropy
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()

        ## Update the learning rate.
        if phase == "train":
           scheduler.step()

        epoch_loss = running_loss / len(dataloader[phase])
        if phase == "train":
            writer.add_scalar('training loss', epoch_loss, epoch * len(dataloader["train"]))
        elif phase == "val":
            writer.add_scalar('val loss', epoch_loss, epoch * len(dataloader["val"]))

        print('{} Epoch Loss: {:.4f}'.format(phase, epoch_loss))



print('Finished Training')

#Saving trained model and loading model.
PATH = './model/firstmodel'
torch.save(net.state_dict(), PATH)
net = Net()
#net.load_state_dict(torch.load(PATH))




