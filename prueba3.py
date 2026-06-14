"""
Clasificador de Melanoma con CNN - Optimizado para RTX 3090
Versión 2.0 - Corregido para Windows y datasets pequeños
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast  # API actualizada
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import json
import argparse
from tqdm import tqdm
import os
from collections import Counter
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================= CONFIGURACIÓN =======================
CONFIG = {
    "imagen_size": 224,
    "batch_size": 16,            # Reducido para más iteraciones por época
    "epochs": 100,
    "learning_rate": 0.0005,     # Reducido para estabilidad
    "min_lr": 1e-6,
    "dropout": 0.5,              # Aumentado para regularización
    "weight_decay": 1e-4,
    "patience": 15,
    "train_path": os.path.join(SCRIPT_DIR, "dataset", "train"),
    "test_path": os.path.join(SCRIPT_DIR, "dataset", "test"),
    "modelo_path": os.path.join(SCRIPT_DIR, "modelo_melanoma_best.pth"),
    "num_workers": 0,            # 0 para Windows (evita problemas de multiprocessing)
    "pin_memory": True,
    "use_amp": True,
}

# ======================= SETUP GPU (LAZY) =======================
_device = None

def get_device():
    """Obtiene el dispositivo de forma lazy (solo cuando se necesita)"""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            _device = torch.device("cpu")
    return _device

def print_gpu_info():
    """Imprime info de GPU (solo llamar desde main)"""
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"✓ CUDA: {torch.version.cuda}")
    else:
        print("⚠ Usando CPU")

# ======================= DATA AUGMENTATION =======================
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((CONFIG["imagen_size"] + 32, CONFIG["imagen_size"] + 32)),
        transforms.RandomCrop(CONFIG["imagen_size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((CONFIG["imagen_size"], CONFIG["imagen_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ======================= MODELO CNN =======================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MelanomaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ======================= MÉTRICAS =======================
def calculate_metrics(y_true, y_pred, class_names):
    """Calcula métricas con nombres de clase correctos"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Encontrar índice de melanoma
    melanoma_idx = class_names.index('melanoma') if 'melanoma' in class_names else 1
    
    # Métricas binarias (melanoma = positivo)
    tp = np.sum((y_pred == melanoma_idx) & (y_true == melanoma_idx))
    tn = np.sum((y_pred != melanoma_idx) & (y_true != melanoma_idx))
    fp = np.sum((y_pred == melanoma_idx) & (y_true != melanoma_idx))
    fn = np.sum((y_pred != melanoma_idx) & (y_true == melanoma_idx))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


# ======================= ENTRENAMIENTO =======================
def get_class_weights(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total = len(targets)
    weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    return torch.tensor([weights[i] for i in range(len(weights))], dtype=torch.float32)


def entrenar():
    device = get_device()
    
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO - Clasificador de Melanoma")
    print("="*60)
    
    # Datasets
    train_dataset = datasets.ImageFolder(CONFIG["train_path"], transform=get_train_transforms())
    test_dataset = datasets.ImageFolder(CONFIG["test_path"], transform=get_test_transforms())
    
    class_names = train_dataset.classes
    print(f"\n📁 Clases: {class_names}")
    print(f"   Índice melanoma: {class_names.index('melanoma')}")
    print(f"📊 Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    # Contar por clase
    train_counts = Counter([s[1] for s in train_dataset.samples])
    print(f"   Distribución train: {dict(zip(class_names, [train_counts[i] for i in range(len(class_names))]))}")
    
    # Class weights
    class_weights = get_class_weights(train_dataset).to(device)
    print(f"⚖️  Pesos: {dict(zip(class_names, class_weights.tolist()))}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )
    
    print(f"📦 Batches por época: {len(train_loader)}")
    
    # Modelo
    model = MelanomaCNN(num_classes=2).to(device)
    print(f"\n🧠 Modelo en {device}")
    print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Mixed precision
    scaler = GradScaler('cuda') if CONFIG["use_amp"] and device.type == 'cuda' else None
    
    # Early stopping (basado en F1)
    early_stopping = EarlyStopping(patience=CONFIG["patience"])
    
    best_f1 = 0.0
    
    print(f"\n🚀 Entrenando ({CONFIG['epochs']} épocas máx)")
    print("-" * 60)
    
    for epoch in range(CONFIG["epochs"]):
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{CONFIG['epochs']}", ncols=100)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(train_labels, train_preds, class_names)
        
        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if scaler:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(test_loader)
        val_metrics = calculate_metrics(val_labels, val_preds, class_names)
        
        # Update scheduler
        scheduler.step(val_metrics['f1'])
        
        # Print
        print(f"\n📈 Época {epoch+1}:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Train Acc:  {train_metrics['accuracy']*100:.1f}% | Val Acc: {val_metrics['accuracy']*100:.1f}%")
        print(f"   Val Precision: {val_metrics['precision']*100:.1f}% | Recall: {val_metrics['recall']*100:.1f}% | F1: {val_metrics['f1']*100:.1f}%")
        print(f"   Confusion: TP={val_metrics['tp']} TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']}")
        
        # Guardar mejor modelo
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'class_names': class_names,
                'config': CONFIG,
            }, CONFIG["modelo_path"])
            print(f"   💾 Mejor modelo guardado (F1: {best_f1*100:.1f}%)")
        
        # Early stopping
        if early_stopping(val_metrics['f1']):
            print(f"\n⏹️  Early stopping en época {epoch+1}")
            break
        
        print("-" * 60)
    
    # Guardar clases
    with open("clases.json", "w") as f:
        json.dump(class_names, f)
    
    print("\n" + "="*60)
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Mejor F1: {best_f1*100:.1f}%")
    print("="*60)


# ======================= PREDICCIÓN =======================
def cargar_modelo():
    device = get_device()
    model = MelanomaCNN(num_classes=2)
    
    checkpoint = torch.load(CONFIG["modelo_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    class_names = checkpoint.get('class_names', ['melanoma', 'sano'])
    print(f"✓ Modelo cargado (F1: {checkpoint['best_f1']*100:.1f}%)")
    return model, class_names


def predecir_img(path):
    device = get_device()
    model, class_names = cargar_modelo()
    transform = get_test_transforms()
    
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if CONFIG["use_amp"] and device.type == 'cuda':
            with autocast('cuda'):
                outputs = model(img_tensor)
        else:
            outputs = model(img_tensor)
        
        probs = F.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    resultado = class_names[pred_idx]
    
    print("\n" + "="*50)
    print(f"📷 Imagen: {os.path.basename(path)}")
    print(f"🔍 Predicción: {resultado.upper()}")
    print(f"📊 Confianza: {confidence*100:.1f}%")
    print("\nProbabilidades:")
    for i, clase in enumerate(class_names):
        prob = probs[0][i].item()
        bar = "█" * int(prob * 30)
        emoji = "🔴" if clase == "melanoma" else "🟢"
        print(f"  {emoji} {clase:10s}: {prob*100:5.1f}% {bar}")
    print("="*50)
    
    return resultado, confidence


def predecir_batch(folder_path):
    device = get_device()
    model, class_names = cargar_modelo()
    transform = get_test_transforms()
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in extensions]
    
    if not images:
        print("❌ No se encontraron imágenes")
        return
    
    print(f"\n📁 Procesando {len(images)} imágenes...")
    results = []
    
    for img_name in tqdm(images, ncols=80):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_idx = outputs.argmax(dim=1).item()
                confidence = probs[0][pred_idx].item()
            
            results.append({
                'imagen': img_name,
                'prediccion': class_names[pred_idx],
                'confianza': confidence,
                'prob_melanoma': probs[0][class_names.index('melanoma')].item()
            })
        except Exception as e:
            print(f"⚠ Error en {img_name}: {e}")
    
    # Resultados
    print("\n" + "="*60)
    melanoma_count = sum(1 for r in results if r['prediccion'] == 'melanoma')
    print(f"📊 Resumen: {melanoma_count} melanoma / {len(results) - melanoma_count} sano")
    
    # Ordenar por probabilidad de melanoma (más probable primero)
    results.sort(key=lambda x: x['prob_melanoma'], reverse=True)
    
    print(f"\n📋 Detalle (ordenado por riesgo):")
    for r in results:
        emoji = "🔴" if r['prediccion'] == 'melanoma' else "🟢"
        print(f"  {emoji} {r['imagen'][:40]:40s} | {r['prediccion']:8s} | {r['confianza']*100:5.1f}%")
    
    return results


# ======================= MENÚ =======================
def menu():
    print("\n" + "="*50)
    print("  CLASIFICADOR DE MELANOMA - CNN v2.0")
    print_gpu_info()
    print("="*50)
    
    while True:
        print("\n📋 MENÚ:")
        print("  1. Entrenar modelo")
        print("  2. Predecir imagen")
        print("  3. Predecir carpeta")
        print("  4. Info del modelo")
        print("  5. Salir")
        
        op = input("\n👉 Opción: ").strip()
        
        if op == "1":
            entrenar()
        elif op == "2":
            path = input("📷 Ruta imagen: ").strip().strip('"')
            if os.path.exists(path):
                predecir_img(path)
            else:
                print("❌ Archivo no encontrado")
        elif op == "3":
            folder = input("📁 Ruta carpeta: ").strip().strip('"')
            if os.path.isdir(folder):
                predecir_batch(folder)
            else:
                print("❌ Carpeta no encontrada")
        elif op == "4":
            if os.path.exists(CONFIG["modelo_path"]):
                ckpt = torch.load(CONFIG["modelo_path"], map_location='cpu', weights_only=False)
                print(f"\n📊 Modelo guardado:")
                print(f"   Época: {ckpt['epoch']+1}")
                print(f"   F1 Score: {ckpt['best_f1']*100:.1f}%")
                print(f"   Clases: {ckpt.get('class_names', 'N/A')}")
            else:
                print("❌ No hay modelo guardado")
        elif op == "5":
            print("\n👋 ¡Hasta luego!")
            break


# ======================= MAIN =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificador de Melanoma v2.0")
    parser.add_argument("--entrenar", action="store_true")
    parser.add_argument("--predecir", type=str)
    parser.add_argument("--batch", type=str)
    args = parser.parse_args()
    
    if args.entrenar:
        print_gpu_info()
        entrenar()
    elif args.predecir:
        predecir_img(args.predecir)
    elif args.batch:
        predecir_batch(args.batch)
    else:
        menu()