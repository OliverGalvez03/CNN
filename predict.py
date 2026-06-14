import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

        conv_output = ((IMAGE_SIZE - 2) - 2) // 2
        self.fc1 = nn.Linear(64 * conv_output * conv_output, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
model.load_state_dict(torch.load("modelo_melanoma.pth", map_location="cpu"))
model.eval()

def predecir_imagen(ruta_img):
    imagen = Image.open(ruta_img).convert("RGB")
    img_tensor = transform(imagen).unsqueeze(0)  # batch de 1

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    clases = ["melanoma", "sano"]
    return clases[pred.item()]

ruta = "img1.jpg"
resultado = predecir_imagen(ruta)
print(f"Clasificacion de imagen: {resultado}")
