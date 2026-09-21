
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import cv2
import torch.nn.functional as F

from utils import evaluar_benchmark





CONFIG = {
    
    "batch": 1,  
    "num_clases": 21,
    "img_size": 512,
    "seed": 42,
    "data": "../data/",
    "checkpoint_dir": "../checkpoints/voc/",
    "logs_dir": "../logs/voc/",
    "mem_dir": "../memoria/benchmark1/",
    "res_data": "../res_data/benchmark1/resultados_voc.csv"
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# Crear carpetas necesarias
#os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
#os.makedirs(CONFIG["logs_dir"], exist_ok=True)
#os.makedirs(CONFIG["mem_dir"], exist_ok=True)


VOC_CLASES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Dataset
class Dataset_VOC(Dataset):
    def __init__(self, root, conjunto="train", transform=None):
        self.dataset = VOCSegmentation(
            root=root, year="2012", image_set=conjunto, download=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indice: int):
        image, mask = self.dataset[indice]
        image = np.array(image)
        mask = np.array(mask)

        # Albumentations exige uint8 para máscaras
        mask = mask.astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Conversión segura a tensor long (independiente de si el transform ya devolvió tensor)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        return image, mask

#
transform_test = A.Compose([
    A.PadIfNeeded(
        min_height=None, min_width=None,
        pad_height_divisor=16, pad_width_divisor=16,
        border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255
    ),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

test_data = Dataset_VOC(root=CONFIG["data"], conjunto="val", transform=transform_test)

test_loader = DataLoader(
    test_data,
    batch_size=CONFIG["batch"],
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Declarar modelos sin pesos inicializados.
unet_val = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=CONFIG["num_clases"]
)
def forward_unet_val(model, images):
    out = model(images)
    if out.shape[-2:] != images.shape[-2:]:
        out = F.interpolate(out, size=images.shape[-2:], 
                           mode="bilinear", align_corners=False)
    return out
deeplabv3plus_val = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=CONFIG["num_clases"]
)
def forward_deeplabv3plus_val(model, images):
    out = model(images)
    if out.shape[-2:] != images.shape[-2:]:
        out = F.interpolate(out, size=images.shape[-2:],
                           mode="bilinear", align_corners=False)
    return out
config_s = SegformerConfig.from_pretrained(
    "nvidia/mit-b2",
    num_labels=CONFIG["num_clases"],
    ignore_mismatched_sizes=True
)
segformer_val = SegformerForSemanticSegmentation(config_s)
def forward_segformer_val(model, images):
    outputs = model(pixel_values=images)
    return F.interpolate(
        outputs.logits,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False
    )
#----------------------------------------------------




if __name__=="__main__":


    print("\n\nIniciando evaluación\n\n")


    voc_models = {
        "unet": (unet_val, forward_unet_val),
        "deeplabv3plus": (deeplabv3plus_val, forward_deeplabv3plus_val),
        "segformer": (segformer_val, forward_segformer_val)
    }

    df_voc = evaluar_benchmark(csv_path=CONFIG["res_data"], modelos_dict=voc_models, indice=1, 
                               loader=test_loader, CONFIG=CONFIG, ignore_index=255, class_names=VOC_CLASES)



    print("\n\nEvaluación finalizada\n\n")