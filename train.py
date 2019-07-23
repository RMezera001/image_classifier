# Imports
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import tensorflow as tf
import argparse

# set up argparse commands
#ref https://code-maven.com/slides/python-programming/argparse-boolean
parser = argparse.ArgumentParser()
parser.add_argument(action='store', dest='data_dir', help='The Directory of folder containing all images')
parser.add_argument('--save_dir',   help='Location for checkpoint save')
parser.add_argument('--arch',   help='Pretrained model archetype')
parser.add_argument('--learning_rate',   help='Learn rate of model')
parser.add_argument('--hidden_units',   help='Units in hidden layer')
parser.add_argument('--epochs',   help='Number of ephochs for training')
parser.add_argument('--gpu', help='Print more data',
    action='store_true')
args = parser.parse_args()

#show settings for training
print('Data Directory: '+ args.data_dir)

if args.save_dir == None:
    print('No  custom save directory chosen')
    print('checkpoint will be saved as: checkpoint.pth')
    save_dir = 'checkpoint.pth'
else:
    print('Save Directory: '+args.save_dir)
    save_dir = str(args.save_dir)+'/checkpoint.pth'
    print('File will be saved as: '+save_dir)
    
if args.arch == None:
    print('No custom pretrained model chosen')
    print('Default pretrained model: vgg16')
    model = models.vgg16(pretrained = True)
else:
    print('Pretrained Model Option: '+args.arch)
    if args.arch == 'vgg11':
        model = models.vgg11(pretrained = True)    
    if args.arch == 'vgg11_bn':
        model = models.vgg11_bn(pretrained = True)
    if args.arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    if args.arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained = True)
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained = True)        
    if args.arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained = True)        
    if args.arch == 'vgg19':
        model = models.vgg19(pretrained = True)        
    if args.arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained = True)
    print(parser.parse_args().arch)
    
if args.learning_rate == None:
    print('No custom learning rate chosen')
    print('Default learning rate set: 0.003')
    learning_rate = 0.003    
else:
    print('Learning Rate: '+args.learning_rate)
    learning_rate = float(args.learning_rate)
    
if args.hidden_units == None:
    print('No custom hidden units chosen')
    print('Default hidden units set: 4096')
    hidden_units = 4096
else:
    print('Hidden Units Option: '+args.hidden_units)
    hidden_units = int(args.hidden_units)
    
if args.epochs == None:
    print('No custom epochs chosen')
    print('Default epochs set: 5')
    epochs = 5    
else:
    print('Epochs Option: '+args.epochs)
    epochs = int(args.epochs)
       
print('GPU Status: '+ str(args.gpu))


# set data directory
data_dir = format(parser.parse_args().data_dir)
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# test_transforms will be used for validatation and train sets
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# load label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# train classifier using ReLU and dropout
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.02)),  
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))   
model.classifier = classifier
criterion = nn.NLLLoss()
# Only train the classifier parameters, Feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# use GPU if available
device = torch.device('cuda' if str(args.gpu)=='True' else 'cpu')
model.to(device)

# Ref Udacity Excercise 8 transfer learning

steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs,labels in trainloader:
        steps+=1
        #move input and label tensors to the default device
        inputs,labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss+= loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs,labels in validloader:
                    inputs,labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps,labels)
                    
                    valid_loss+= batch_loss.item()
                    
                    #calculate accuracy
                    
                    ps=torch.exp(logps)
                    top_p,top_class = ps.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()         

# TODO: Do validation on the test set
test_loss = 0
accuracy = 0
with torch.no_grad():
    for inputs,labels in testloader:
        inputs,labels = inputs.to(device),labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps,labels)

        test_loss+= batch_loss.item()

        #calculate accuracy
        ps=torch.exp(logps)
        top_p,top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy/len(testloader):.3f}")

#set class to index checkpoint
model.class_to_idx = train_data.class_to_idx

# TODO: Save the checkpoint 
#Ref Udacity excercise 6
checkpoint = {'input_size': 25088,
              'hidden_layer': hidden_units,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'model_arch': args.arch}

torch.save(checkpoint, save_dir)