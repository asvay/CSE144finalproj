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