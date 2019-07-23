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
parser.add_argument(action='store', dest='image_dir', help='The Directory to image')
parser.add_argument(action='store', dest='checkpoint_path', help='Checkpoint path')
parser.add_argument('--top_k',   help='Number of Top Percent')
parser.add_argument('--category_names',   help='Dictionary of categories to names')
parser.add_argument('--gpu', help='Print more data',
    action='store_true')
args = parser.parse_args()

#show settings for training
print('Image Directory: '+ args.image_dir)
print('Checkpoint Directory: '+ args.checkpoint_path)

if args.top_k == None:
    print('No custom setting chosen for top_k')
    print('Default top_k = 5')
    top_k = 5
else:
    print('Top_k: '+args.top_k)
    top_k = int(args.top_k)

if args.category_names == None:
    print('Category names not chosen and a number will be used to represent the flower.')
    cat_status = False
else:
    print('Category names chosen using file: '+args.category_names)
    cat_status = True
    # load label mapping
    with open(str(args.category_names), 'r') as f:
        cat_to_name = json.load(f)    

print('GPU Status: '+ str(args.gpu))
    
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
#Load checkpoint
checkpoint = torch.load(args.checkpoint_path)

# load pretrained model used in training
if str(checkpoint['model_arch']) == 'vgg11':
    model = models.vgg11(pretrained = True)    
if str(checkpoint['model_arch']) == 'vgg11_bn':
    model = models.vgg11_bn(pretrained = True)
if str(checkpoint['model_arch']) == 'vgg13':
    model = models.vgg13(pretrained = True)
if str(checkpoint['model_arch']) == 'vgg13_bn':
    model = models.vgg13_bn(pretrained = True)
if str(checkpoint['model_arch']) == 'vgg16':
    model = models.vgg16(pretrained = True)        
if str(checkpoint['model_arch']) == 'vgg16_bn':
    model = models.vgg16_bn(pretrained = True)        
if str(checkpoint['model_arch']) == 'vgg19':
    model = models.vgg19(pretrained = True)        
if str(checkpoint['model_arch']) == 'vgg19_bn':
    model = models.vgg19_bn(pretrained = True)
if str(checkpoint['model_arch']) == 'None':
    model = models.vgg16(pretrained = True)

# load classifier/state dict/ class to idx
hidden_units = int(checkpoint['hidden_layer'])
classifier  = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.02)),  
    ('fc2', nn.Linear(hidden_units,102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
criterion = nn.NLLLoss()

model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']
print('model loaded')


#resize/crop image and return the image, image width, and image hight after the resize
# to be used in 'process_image
def resize_crop_image(image):
    '''scales and crops image
    returns: resize_image, resize_width, resize_height
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    #use shortest side to resize image
    im = Image.open(image)
    width, height = im.size
    if width < height:
        denominator = width/256
        resize_width = 256
        resize_height = int(height/denominator)
        resize_image = im.resize((256,resize_height), Image.ANTIALIAS)
        return resize_image, resize_width, resize_height
    else:
        denominator = height/256
        resize_width = int(width/denominator)
        resize_height = 256
        resize_image = im.resize((resize_width,256), Image.ANTIALIAS)
        return resize_image, resize_width, resize_height

# takes image a numpy array    
def process_image(image):
    ''' Normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
    resize_image, resize_width, resize_height = resize_crop_image(image)

    #now we will use resize image/width/height to crop image
    mid_width = resize_width/2
    mid_height = resize_height/2
    #coordinates (x1,y1,x2,y2)
    coords = (mid_width - 112, mid_height - 112, mid_width + 112, mid_height + 112)
    cropped_image = resize_image.crop(coords)
    
    #convert to np array
    np_image = np.array(cropped_image)/255
    #set mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mean) / std
    # transpose 3rd channel to 1st
    final_image = normalized_image.transpose((2,0,1))
    
    return final_image    

# takes numpy array and returns image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def return_label(loc):
    '''used to return class number given dict colation
    '''
    for label, location in model.class_to_idx.items():
        if location == loc:
            return label
        
# predict image and return probabilites and classes      
def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file  
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Change image to tensor
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0)
    
    # move to GPU if available
    image, model = image.to(device), model.to(device)

    # Eval image using model
    logps = model(image)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(top_k)
    
    #get labels from label location in dictionary
    labels=[]
    for x in top_class[0][0:top_k]:
        labels.append(return_label(x))
    
    return top_p, labels
probs, classes = predict(args.image_dir, model)

# return integer or names of flowers depending on if --category_names cat_to_name.json was used
# Convert labels number to cat name

if cat_status == False:
    top_cat = classes
if cat_status == True:
    top_cat=[]
    for x in classes:
        top_cat.append(cat_to_name[x])
        
# put probabilites into np array
probs = probs.cpu()

#change probs from tensor to np.array
probs_list = []
for x in np.array(probs.detach().numpy())[0]:
    probs_list.append(x)
    
#Print results  
for x in range(len(probs_list)):
    print('Prediction #'+str(x+1)+': Category: '+str(top_cat[x])+' Probability: '+str(probs_list[x]))
