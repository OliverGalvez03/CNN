import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "imagen_size": 128,
    "batch_size": 32,
    "epocas": 10,
    "learning_rate": 0.001,
    "dropout": 0.25,
    "train_path": os.path.join(SCRIPT_DIR, "dataset", "train"),
    "test_path": os.path.join(SCRIPT_DIR, "dataset", "test"),
    "modelo_path": os.path.join(SCRIPT_DIR, "modelo_cnn.pth")
}

def get_transforms(es_entrenamiento=False):
    if es_entrenamiento:
        return transforms.Compose([
            transforms.Resize((CONFIG["imagen_size"], CONFIG["imagen_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG["imagen_size"], CONFIG["imagen_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class MelanomaCNN(nn.Module):
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        tamaño_final = CONFIG["imagen_size"] // 8
        self.fc1 = nn.Linear(128 * tamaño_final * tamaño_final, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def cargar_datos():
    if not os.path.exists(CONFIG["train_path"]):
        print(f"No existe la ruta {CONFIG['train_path']}")
        return None, None, None
    
    train_dataset = ImageFolder(CONFIG["train_path"], transform=get_transforms(True))
    test_dataset = ImageFolder(CONFIG["test_path"], transform=get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    return train_loader, test_loader, train_dataset.classes

def entrenar_modelo():
    train_loader, test_loader, clases = cargar_datos()
    if train_loader is None:
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = MelanomaCNN().to(device)
    optimizer = optim.Adam(modelo.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    mejor_accuracy = 0

    for epoch in range(CONFIG["epocas"]):
        modelo.train()
        total_loss = 0
        correctos = 0
        total = 0
        
        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            optimizer.zero_grad()
            predicciones = modelo(imagenes)
            loss = criterion(predicciones, etiquetas)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = predicciones.max(1)
            correctos += pred.eq(etiquetas).sum().item()
            total += etiquetas.size(0)

        test_acc = evaluar_modelo(modelo, test_loader, device)
        
        if test_acc > mejor_accuracy:
            mejor_accuracy = test_acc
            guardar_modelo(modelo, clases)
        
        print(f"Epoca {epoch+1}/{CONFIG['epocas']} │ Loss: {total_loss/len(train_loader):.4f} ││ Test Acc: {test_acc:.1f}%")

def evaluar_modelo(modelo, loader, device):
    modelo.eval()
    correctos = 0
    total = 0
    
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            predicciones = modelo(imagenes)
            _, pred = predicciones.max(1)
            correctos += pred.eq(etiquetas).sum().item()
            total += etiquetas.size(0)
    return 100. * correctos / total

def guardar_modelo(modelo, clases):
    torch.save({
        "model_state_dict": modelo.state_dict(),
        "clases": clases,
        "config": CONFIG
    }, CONFIG["modelo_" \
    "path"])

def cargar_modelo_entrenado():
    if not os.path.exists(CONFIG["modelo_path"]):
        print("Modelo no encontrado. Entrena primero el modelo.")
        return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CONFIG["modelo_path"], map_location=device)
    modelo = MelanomaCNN()
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()
    return modelo, checkpoint["clases"]

# ------------------------------------------------------------------------------------------------
def mostrar_metricas_completas():
    modelo, clases = cargar_modelo_entrenado()
    if modelo is None: return
    
    _, test_loader, _ = cargar_datos()
    if test_loader is None: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_labels = []

    print("\nGenerando Matriz de Confusion")
    
    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            imagenes = imagenes.to(device)
            outputs = modelo(imagenes)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(etiquetas.numpy())

    # 1. Matriz de Confusion
    cm = confusion_matrix(all_labels, all_preds)
    
    # Visualización gráfica con Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=clases, yticklabels=clases, cmap='Blues')
    plt.xlabel('Prediccion del Modelo')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusion: Melanoma')
    plt.show()

    print("\n" + "="*60)
    print("Reporte de Clasificacion (Precision, Recall, F1-Score)")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=clases))
    print("="*60 + "\n")
# -----------------------------------------------------------------------------------------------------

def predecir_imagen(ruta_imagen):
    modelo, clases = cargar_modelo_entrenado()
    if modelo is None: return
    
    if not os.path.exists(ruta_imagen):
        print(f"La imagen {ruta_imagen} no existe.")
        return
    
    imagen = Image.open(ruta_imagen).convert("RGB")
    transform = get_transforms(False)
    img_tensor = transform(imagen).unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = modelo(img_tensor)
        probabilidades = F.softmax(output, dim=1)
        confianza, prediccion = torch.max(probabilidades, 1)
    
    print(f"Clasificacion: {clases[prediccion.item()]} ({confianza.item()*100:.1f}%)")

def ver_configuracion():
    for key, value in CONFIG.items():
        print(f"{key}: {value}")

def mostrar_menu():
    print("""
1. Entrenar modelo
2. Predecir imagen
3. Ver configuracion
4. Matriz de Confusion
5. Salir
""")

def main_menu():
    while True:
        mostrar_menu()
        opcion = input("Opcion: ").strip()
        
        if opcion == "1":
            entrenar_modelo()
        elif opcion == "2":
            ruta = input("Ruta imagen: ").strip().replace('"', '').replace("'", "")
            predecir_imagen(ruta)
        elif opcion == "3":
            ver_configuracion()
        elif opcion == "4":
            mostrar_metricas_completas()
        elif opcion == "5":
            break
        else:
            print("Opcion no valida.")

def configurar_argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrenar", action="store_true")
    parser.add_argument("--predecir", type=str)
    parser.add_argument("--metricas", action="store_true")
    parser.add_argument("--epocas", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = configurar_argumentos()
    if args.epocas:
        CONFIG["epocas"] = args.epocas
    
    if args.entrenar:
        entrenar_modelo()
    elif args.predecir:
        predecir_imagen(args.predecir)
    elif args.metricas:
        mostrar_metricas_completas()
    else:
        main_menu()