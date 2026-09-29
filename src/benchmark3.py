
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import Cityscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import cv2
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

from utils import evaluar_benchmark


CONFIG = {
                                       
    "batch": 1,  
    "num_clases": 11,
    "seed": 42,
    "data": "../data/",
    "checkpoint_dir": "../checkpoints/liaci/",
    "logs_dir": "../logs/liaci/",
    "mem_dir": "../memoria/benchmark3/",
    "res_data": "../res_data/benchmark3/resultados_liaci.csv"
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

LIACI_CLASES = [
    'Void',                 #void
    'Ship hull',            #Casco de barco
    'Marine growth',        #Crecimiento marino
    'Anode',                #Ánodo
    'Overboard valve',      #Válvula de sesagüe
    'Propeller',            #Hélice
    'Paint peel',           #Desprendimiento de pintura
    'Bilge keel',           #Quilla de balance
    'Defect',               #Defecto
    'Corrosion',            #Corrosión
    'Sea chest grating'     #Rejilla de la toma de mar
]
CLASS_COLORS = [
    (0, 0, 0),              # Void - Negro
    (0, 0, 255),            # Ship hull - Azul
    (0, 128, 0),            # Marine growth - Verde
    (0, 255, 255),          # Anode - Cyan
    (64, 224, 208),         # Overboard valve - Turquesa
    (128, 0, 128),          # Propeller - Purpura
    (255, 0, 0),            # Paint peel - Rojo
    (255, 165, 0),          # Bilge keel - Naranja
    (255, 192, 203),        # Defect - Rosa
    (255, 255, 0),          # Corrosion - Amarillo
    (255, 255, 255),        # Sea chest grating - Blanco
]
cmap_liaci = ListedColormap(np.array(CLASS_COLORS)/255.0)


def convert_color_to_id(mask_rgb):
    """
    Convierte una máscara RGB [H, W, 3] a una máscara de índices [H, W].
    Cada color corresponde a una clase según CLASS_COLORS.
    Los colores no reconocidos se asignan a 0 (Void).
    """
    h, w = mask_rgb.shape[:2]
    # Codificar cada píxel RGB como un único entero: R*256² + G*256 + B
    mask_flat = (mask_rgb[:, :, 0].astype(np.int32) * 256 * 256 +
                 mask_rgb[:, :, 1].astype(np.int32) * 256 +
                 mask_rgb[:, :, 2].astype(np.int32))
    
    # Inicializar todo a 0 (Void)
    mask_id = np.zeros((h, w), dtype=np.uint8)
    
    for idx, (r, g, b) in enumerate(CLASS_COLORS):
        color_code = r * 256 * 256 + g * 256 + b
        mask_id[mask_flat == color_code] = idx
    
    return mask_id

#Dataset Liaci
class Dataset_LIACi(Dataset):
    def __init__(self, root, split_df, conjunto="train", transform=None):
        """
        root: carpeta raíz de LIACi
        split_df: DataFrame del CSV (con columnas 'file_name' y 'split')
        conjunto: "train" o "test"
        """
        
        self.images_dir = os.path.join(root, "images")
        self.seg_dir = os.path.join(root, "segmentation")
        self.transform = transform

        # Filtrar por split, excluye automáticamente cualquier otro valor
        self.archivos = split_df[split_df["split"] == conjunto]["file_name"].tolist()
        
        print(f"Dataset LIACi [{conjunto}]: {len(self.archivos)} imágenes")

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, indice):
        nombre_img = self.archivos[indice]
        
        # Cargar imagen .jpg
        image = np.array(
            Image.open(os.path.join(self.images_dir, nombre_img)).convert("RGB")
        )
        
        # Máscara: mismo nombre base con extensión .png
        nombre_mask = nombre_img.replace(".jpg", ".png")
        mask_rgb = np.array(
            Image.open(os.path.join(self.seg_dir, nombre_mask)).convert("RGB")
        )
        mask = convert_color_to_id(mask_rgb)
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        return image, mask.long()


#
transform_test = A.Compose([
    A.PadIfNeeded(
        min_height=None, min_width=None,
        pad_height_divisor=16, pad_width_divisor=16,
        border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0
    ),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

#####################################################
df = pd.read_csv("../data/LIACi/train_test_split.csv")
test_dataset = Dataset_LIACi(
    root="../data/LIACi",
    split_df=df,
    conjunto="test",
    transform=transform_test
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=CONFIG["batch"],
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
#######################################################

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
##################################################



if __name__=="__main__":

    print("\n\nIniciando evaluación\n\n")

    liaci_models = {
        "unet":(unet_val, forward_unet_val),
        "deeplabv3plus":(deeplabv3plus_val, forward_deeplabv3plus_val),
        "segformer": (segformer_val, forward_segformer_val)
    }

    df_liaci = evaluar_benchmark(csv_path=CONFIG["res_data"], modelos_dict=liaci_models, indice=3,
                                 loader=test_loader, CONFIG=CONFIG, class_names=LIACI_CLASES)



    print("\n\nEvaluación finalizada\n\n")

