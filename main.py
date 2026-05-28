# libraries
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from PIL import Image

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset

SEED = 42
NUM_CLASSES = 100
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

DATA_DIR = "./data"
TRAIN_DIR = "./data/train"
TEST_DIR = "./data/test"
SAMPLE_SUBMISSION = "./data/sample_submission.csv"
CKPT_PATH = "./checkpoints/best_model.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    # same idea from HW2
def build_transforms():
    # resize, augment train images, normalize for pretrained model
def build_dataloaders():
    # load train folders with ImageFolder
    # split into train/val
    # return train_loader, val_loader, class_to_idx
def build_model(num_classes=100):
    # load pretrained ResNet/EfficientNet
    # replace classifier head
def train_one_epoch(model, loader, criterion, optimizer):
    # reused from HW2
@torch.no_grad()
def evaluate(model, loader, criterion):
    # reused from HW2
def train_model():
    # loop epochs
    # save best checkpoint
    # track history
class TestImageDataset(Dataset):
    # loads test images from sample_submission.csv order
@torch.no_grad()
def generate_submission(model, test_loader):
    # predict labels
    # write submission.csv