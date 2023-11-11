
# Basic usage: python train.py data_directory
# bash 
# python train.py data_directory --arch vgg16 --hidden_units 1024 --learn_rate 0.002 --gpu

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
import torchvision.models as models
import numpy as np
import os, random
import json
import signal
from PIL import Image
from contextlib import contextmanager
import requests
import matplotlib
import matplotlib.pyplot as plt
import argparse


def input_args():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a folder of images.')

    # Required argument
    parser.add_argument('dir', type=str, help='path to folder of images')

    # Optional arguments
    parser.add_argument('--arch', type=str, default='resnet50', help='chosen model architecture')
    parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units')
    parser.add_argument('--learn_rate', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('--save', type=str, default='checkpoint.pth', help='checkpoint file name to save')
    parser.add_argument('--gpu', action="store_true", help='use GPU for training')

    # Return parsed arguments
    return parser.parse_args()



# Data preperation wrapped function
def data_load(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_datasets = ImageFolder(train_dir, transform = train_transforms)
    test_datasets= ImageFolder(test_dir, transform = test_transforms)
    valid_datasets= ImageFolder(valid_dir, transform = valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_datasets_loaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    test_datasets_loaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)
    valid_datasets_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)
    
    class_to_idx = train_datasets.class_to_idx
    
    return train_datasets_loaders,valid_datasets_loaders,class_to_idx

def build_model(args, class_to_idx): 
    
    model = getattr(torchvision.models, args.arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if "vgg" in args.arch:
        in_features_of_pretrained = model.classifier[0].in_features
    else:
        in_features_of_pretrained = model.fc.in_features

    nflower_classes = len(class_to_idx)
    
    # Hyperparameters
    hidden_units = args.hidden_units
    learning_rate = args.learn_rate

    # Adjusted based on user-defined hyperparameters
    clasificador = nn.Sequential(
        nn.Linear(in_features=in_features_of_pretrained, out_features=hidden_units, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=hidden_units, out_features=nflower_classes, bias=True),
        nn.LogSoftmax(dim=1)
    )
    
    if "vgg" in args.arch:
        model.classifier = clasificador
    else:
        model.fc = clasificador
    
    return model




def train_model(args, model, train_datasets_loaders, valid_datasets_loaders, class_to_idx):
    # trains model, saves model to dir, returns True if sucessful 
    # train model
    # criterion
    criterion = nn.NLLLoss()
    # optimizer
        # Check if the model has 'classifier' or 'fc' attribute
    if hasattr(model, 'classifier'):
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learn_rate)
    elif hasattr(model, 'fc'):
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learn_rate)
    else:
        raise AttributeError("Model does not have 'classifier' or 'fc' attribute.")

    # decide device depending on user arguments and device availability
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'
    print("Using {} to train model.".format(device))

    # move model to selected device
    model.to(device)

    # init variables for tracking loss/steps etc.
    print_every = 20

    # for each epoch

    for e in range(args.epochs):
        model.train()
        running_train_loss = 0

        for step, (images, labels) in enumerate(train_datasets_loaders, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

            if step % print_every == 0 or step == len(train_datasets_loaders):
                print(f"Epoch: {e+1}/{args.epochs} Batch % Complete: {100 * step / len(train_datasets_loaders):.2f}%")

        model.eval()
        with torch.no_grad():
            running_valid_loss = 0
            running_accuracy = 0

            for images, labels in valid_datasets_loaders:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
                running_valid_loss += valid_loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        average_train_loss = running_train_loss / len(train_datasets_loaders)
        average_valid_loss = running_valid_loss / len(valid_datasets_loaders)
        accuracy = running_accuracy / len(valid_datasets_loaders)

        print(f"Train Loss: {average_train_loss:.3f}")
        print(f"Valid Loss: {average_valid_loss:.3f}")
        print(f"Accuracy: {accuracy * 100:.3f}%")
        
    #save model

    model.class_to_idx = class_to_idx
    
    if hasattr(model, 'classifier'):
            checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'vgg_type': args.arch
                 }
     
    elif hasattr(model, 'fc'):
            checkpoint = {'fc': model.fc,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'resnet_type': args.arch
                 }
    else:
        raise AttributeError("Model does not have 'classifier' or 'fc' attribute.")    
 
#change ur directory
    torch.save(checkpoint, os.path.join(args.save, "/workspace/home/ImageClassifier/checkpoint.pth"))
    print("model saved to {}".format(os.path.join(args.save, "checkpoint.pth")))
    return True
    
    
def main():

    # Call argument parser function
    args = input_args()
    # Load data with arg parser data directory argument
    train_datasets_loaders, valid_datasets_loaders, class_to_idx = data_load(args.dir)
    # Load model, save data
    modelo = build_model(args, class_to_idx)
    train_model(args, modelo, train_datasets_loaders, valid_datasets_loaders, class_to_idx)
 
       
# Call to main function to run the program
if __name__ == '__main__':
  
    main()
    