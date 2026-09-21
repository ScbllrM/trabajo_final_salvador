import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Cityscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import cv2
import torch.nn.functional as F

from utils import evaluar_benchmark


