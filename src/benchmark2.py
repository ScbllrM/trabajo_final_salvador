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


CONFIG = {
    
    "batch": 1,
    "num_clases": 19,
    "seed": 42,
    "data": '../data/cityscapes',
    "checkpoint_dir": "../checkpoints/cityscapes/",
    "logs_dir": "../logs/cityscapes/",
    "mem_dir": "../memoria/benchmark2/",
    "res_data": "../res_data/benchmark2/resultados_cityscapes.csv"
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


city_clases = [

    "license plate"
    "unlabeled"
    "ego vehicle"
    "rectification border"
    "out of roi"
    "static"
    "dynamic"
    "ground"
    "road"
    "sidewalk"
    "parking"
    "rail track"
    "building"
    "wall"
    "fence"
    "guard rail"
    "bridge"
    "tunnel"
    "pole"
    "polegroup"
    "traffic light"
    "traffic sign"
    "vegetation"
    "terrain"
    "sky"
    "person"
    "rider"
    "car"
    "truck"
    "bus"
    "caravan"
    "trailer"
    "train"
    "motorcycle"
    "bicycle"
]



##
ID_TO_TRAINID = {
    -1: 255,   # license plate
    0: 255,  # unlabeled
    1: 255,  # ego vehicle
    2: 255,  # rectification border
    3: 255,  # out of roi
    4: 255,  # static
    5: 255,  # dynamic
    6: 255,  # ground
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255,  # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18    # bicycle
}
##
LUT = np.full(34, 255, dtype=np.uint8)
for id_original, train_id in ID_TO_TRAINID.items():
    if id_original >= 0:
        LUT[id_original] = train_id


def convert_mask_id(mask):
    """Convierte máscara con IDs originales de Cityscapes a trainIds."""
    mask = np.asarray(mask, dtype=np.int32)
    
    mask = np.clip(mask, 0, 33)  # protege contra valores inesperados
    return LUT[mask]
#Dataset
class Dataset_Cityscapes(Dataset):

    def __init__(self, conjunto="train", transform=None):
        self.dataset = Cityscapes(
            root=CONFIG["data"],
            split=conjunto,
            mode="fine",
            target_type="semantic"
        )
        self.transform=transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, indice:int):
        image, mask = self.dataset[indice]

        image = np.array(image)
        mask = np.array(mask)

        mask = convert_mask_id(mask)
        mask = mask.astype(np.uint8)

        if self.transform:
            t = self.transform(image=image, mask=mask)
            image = t["image"]
            mask = t["mask"].long()
        return image, mask


transform_test = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

test_data = Dataset_Cityscapes(conjunto="val", transform=transform_test)

test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

#________________________________________
#Declarar modelos sin pesos inicializados
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
#______________________________________________











if __name__=="__main__":

    print("\n\nIniciando evaluación\n\n")

    city_models = {
        "unet": (unet_val,forward_unet_val),
        "deeplabv3plus": (deeplabv3plus_val, forward_deeplabv3plus_val),
        "segformer": (segformer_val, forward_segformer_val)
    }

    df_city = evaluar_benchmark(csv_path=CONFIG["res_data"], modelos_dict=city_models, indice=2,
                                loader=test_loader, CONFIG=CONFIG, ignore_index=255, class_names=city_clases)



    print("\n\nEvaluación finalizada\n\n")





