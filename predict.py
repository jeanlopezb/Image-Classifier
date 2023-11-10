
#promt python3 predict.py  /workspace/home/ImageClassifier/flowers/test/1/image_06764.jpg
#change the image_directory to predict other image.
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import json
import argparse


def input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('single_image', type=str, help='path to the image that should be predicted')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='checkpoint file name')
    parser.add_argument('--topk', type=int, default=5, help='specify number of output predictions')
    parser.add_argument('--flower_map', type=str, default='cat_to_name.json', help='class to flower-species map')    
    parser.add_argument('--gpu', action='store_true', default=False, help='gpu training option')

    # Returns argument parser
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if 'vgg_type' in checkpoint:
        # Load VGG model
        if checkpoint['vgg_type'] == "vgg11":
            model = torchvision.models.vgg11(pretrained=True)
        elif checkpoint['vgg_type'] == "vgg13":
            model = torchvision.models.vgg13(pretrained=True)
        elif checkpoint['vgg_type'] == "vgg16":
            model = torchvision.models.vgg16(pretrained=True)
        elif checkpoint['vgg_type'] == "vgg19":
            model = torchvision.models.vgg19(pretrained=True)
    elif 'resnet_type' in checkpoint:
        # Load ResNet model
        if checkpoint['resnet_type'] == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported model type in the checkpoint")

    # Rebuild the classifier/fc based on the loaded checkpoint
    if 'classifier' in checkpoint:
        model.classifier = checkpoint['classifier']
    elif 'fc' in checkpoint:
        model.fc = checkpoint['fc']

    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image_path)
    

    # TODO: Process a PIL image for use in a PyTorch model
    
    # reize
    pil_image = Image.open(image_path)

    # Resize
    pil_image = pil_image.resize((256, 256))

    # Center crop
    width, height = pil_image.size
    new_width, new_height = 224, 224
    
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Crop the center of the image
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert color channel from 0-255 to 0-1
    np_image = np.array(pil_image) / 255.0

    # Normalize for model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose color channels to the first dimension
    np_image = np_image.transpose((2, 0, 1))

    # Convert to Float Tensor
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)

    return tensor


def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # move to device
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))
        
    ps, top_classes = ps.topk(topk, dim=1)
    
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]

    # returning both as lists instead of torch objects for simplicity
    return ps.tolist()[0], predicted_flowers_list


def main(args):
    # load model
    model = load_checkpoint(args.checkpoint_path)

    # decide device depending on user arguments and device availability
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    if args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'

    model = model.to(device)

    # print(model.class_to_index)

    with open(args.flower_map, 'r') as f:
        cat_to_name = json.load(f)

    # predict image
    top_ps, top_classes = predict(args.single_image, model, args.topk, device, cat_to_name)

    print("Predictions:")
    for i in range(args.topk):
          print("#{: <3} {: <25} Prob: {:.2f}%".format(i, top_classes[i], top_ps[i]*100))
            

if __name__ == "__main__":
    args = input_args()
    main(args)          
   
