import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import json
import argparse
from tqdm import tqdm

CONFIG = {
    "imagen_size": 128,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "dropout": 0.25,
    "train_path": "./dataset/train",
    "test_path": "./dataset/test",
    "modelo_path": "modelo_melanoma.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cargar_transforms():
    return transforms.Compose([
        transforms.Resize((CONFIG["imagen_size"], CONFIG["imagen_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        s = CONFIG["imagen_size"]
        self.c1 = nn.Conv2d(3, 32, 3, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.c3 = nn.Conv2d(64, 128, 3, 1)
        self.b1 = nn.BatchNorm2d(32)
        self.b2 = nn.BatchNorm2d(64)
        self.b3 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(CONFIG["dropout"])

        size = (((s - 2) // 2 - 2) // 2 - 2) // 2
        self.fc1 = nn.Linear(128 * size * size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.b2(self.c2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.b3(self.c3(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def entrenar():
    t = cargar_transforms()
    train = datasets.ImageFolder(CONFIG["train_path"], transform=t)
    test = datasets.ImageFolder(CONFIG["test_path"], transform=t)

    train_loader = DataLoader(train, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test, batch_size=CONFIG["batch_size"], shuffle=False)

    model = CNN().to(device)
    opt = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(CONFIG["epocas"]):
        model.train()
        total = 0
        for x, y in tqdm(train_loader, desc=f"Epoca {e+1}/{CONFIG['epocas']}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print("Loss:", total / len(train_loader))
        evaluar(model, test_loader)

    torch.save(model.state_dict(), CONFIG["modelo_path"])
    with open("clases.json", "w") as f:
        json.dump(train.classes, f)
    with open("config.json", "w") as f:
        json.dump(CONFIG, f)
    print("Modelo guardado")

def evaluar(model, loader):
    model.eval()
    c = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            c += (p == y).sum().item()
    acc = c / len(loader.dataset) * 100
    print("Porcentaje de acierto:", f"{acc:.2f}%")

def cargar_modelo():
    model = CNN()
    model.load_state_dict(torch.load(CONFIG["modelo_path"], map_location=device))
    model.to(device)
    model.eval()
    with open("clases.json") as f:
        clases = json.load(f)
    return model, clases

def predecir_img(path):
    model, clases = cargar_modelo()
    transform = cargar_transforms()

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        p = torch.argmax(out).item()

    print("Resultado:", clases[p])

def predecir_carpeta(folder):
    model, clases = cargar_modelo()
    transform = cargar_transforms()

    for img_name in os.listdir(folder):
        if img_name.lower().endswith(("jpg", "png", "jpeg")):
            path = os.path.join(folder, img_name)
            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img)
                p = torch.argmax(out).item()
            print(img_name, "=>", clases[p])

def mostrar_config():
    print(json.dumps(CONFIG, indent=4))

def menu():
    while True:
        print("\n1. Entrenar modelo")
        print("2. Predecir imagen")
        print("3. Predecir carpeta")
        print("4. Ver configuración")
        print("5. Salir")

        op = input("Opción: ")

        if op == "1":
            entrenar()
        elif op == "2":
            p = input("Imagen: ")
            predecir_img(p)
        elif op == "3":
            f = input("Carpeta: ")
            predecir_carpeta(f)
        elif op == "4":
            mostrar_config()
        elif op == "5":
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrenar", action="store_true")
    parser.add_argument("--predecir", type=str)
    parser.add_argument("--carpeta", type=str)
    args = parser.parse_args()

    if args.entrenar:
        entrenar()
    elif args.predecir:
        predecir_img(args.predecir)
    elif args.carpeta:
        predecir_carpeta(args.carpeta)
    else:
        menu()