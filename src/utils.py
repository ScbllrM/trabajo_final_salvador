# Importación de paquetes
import os
import time
import gc
import numpy as np
import pandas as pd
import torch
from torchmetrics import JaccardIndex



#### GPU Info ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
##################


#
def calcular_tamano_modelo(modelo):
    """
    Calcula el tamaño del modelo en MB sumando el tamaño en bytes
    de todos los parámetros y buffers.
    """
    param_size = sum(p.numel() * p.element_size() for p in modelo.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in modelo.buffers())
    total_mb = (param_size + buffer_size) / (1024 ** 2)
    return total_mb



def evaluar_model(nombre, indice, modelo, forward_fn, CONFIG, loader, ignore_index=None, class_names=None):
    
    # Cargar checkpoint entrenados
    checkpoint = torch.load(
        f"{CONFIG['checkpoint_dir']}{nombre}_b{indice}_checkpoint.pth",
        map_location=device
    )

    modelo.load_state_dict(checkpoint["modelo"])   #<- modelo entrenado
    modelo.eval()
    modelo = modelo.to(device)
    
    print(f"{nombre} cargado | mIoU guardado: {checkpoint['mejor_miou']*100:.2f}%")

    #---------------------------------
    miou_fn = JaccardIndex(
        task="multiclass",
        num_classes=CONFIG["num_clases"],
        ignore_index=ignore_index,
        average="macro"
    ).to(device)

    miou_por_clase = JaccardIndex(
        task="multiclass",
        num_classes=CONFIG["num_clases"],
        ignore_index=ignore_index,
        average="none"
    ).to(device)
    #---------------------------------
    tiempos = []
    n_params = sum(p.numel() for p in modelo.parameters()) / 1e6
    
    # Tamaño del modelo en disco/memoria
    tam_mb = calcular_tamano_modelo(modelo)

    # Warmup de GPU
    with torch.no_grad():
        dummy = next(iter(loader))[0][:1].to(device)
        for _ in range(5):
            _ = forward_fn(modelo, dummy)
    torch.cuda.synchronize()

    # Evaluación real
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.long().to(device)
            batch_size = images.shape[0]

            torch.cuda.synchronize()
            inicio = time.time()
            preds = forward_fn(modelo, images).argmax(dim=1)
            torch.cuda.synchronize()
            fin = time.time()

            # Tiempo POR IMAGEN (dividir entre el tamaño del batch)
            tiempos.append((fin - inicio) / batch_size)
            miou_fn.update(preds, masks)
            miou_por_clase.update(preds, masks)

    
    fps = 1.0 / np.mean(tiempos)
    miou = miou_fn.compute().item()
    iou_clase = miou_por_clase.compute().cpu().numpy()


    if class_names is not None:
        iou_dict = {class_names[i]: round(float(iou_clase[i]) * 100, 2)
                    for i in range(len(iou_clase))}
        print(f"    IoU por clase: {iou_dict}")
    else:
        iou_dict = {f"clase_{i}": round(float(v) *100, 2)
                    for i, v in enumerate(iou_clase)}


    # Limpieza VRAM
    modelo.cpu()
    del modelo, miou_fn
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    res =  {
        "Modelo": nombre,
        "mIoU (%)": round(miou * 100, 2),
        "FPS": round(fps, 1),
        "Params (M)": round(n_params, 1),
        "Tamaño (MB)": round(tam_mb, 1)
    }
    res.update({f"IoU_{k}": v for k, v in iou_dict.items()})

    return res

#
def evaluar_benchmark(csv_path, modelos_dict, indice, loader, CONFIG, ignore_index=None, class_names=None):

    resultados = []
    for nombre, (model, forward_fn) in modelos_dict.items():
        res = evaluar_model(nombre=nombre, indice=indice, modelo=model,forward_fn=forward_fn, loader=loader, CONFIG=CONFIG, 
                            ignore_index=ignore_index, class_names=class_names)
        resultados.append(res)

    df=pd.DataFrame(resultados)
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")
    
    return df