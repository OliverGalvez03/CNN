"""
===========================================
    DETECTOR DE MELANOMA - EDICIÓN OLIVER
===========================================
Oliver, este es tu detector de melanoma usando redes neuronales.
Aquí está todo junto: entrenar, probar y predecir.
Agarra tu café y vamos paso a paso.
"""

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
import sys

# ===========================================================================
# CONFIGURACIÓN GENERAL
# ===========================================================================
# Oliver, aquí defines los "botones" de tu red neuronal.
# Si quieres experimentar, estos son los valores que puedes tocar.

CONFIG = {
    "imagen_size": 128,          # tamaño al que se redimensionan las fotos
    "batch_size": 32,            # cuántas imágenes procesa a la vez (como bocados)
    "epochs": 10,                # cuántas vueltas le da al dataset completo
    "learning_rate": 0.001,      # qué tan rápido aprende (muy alto = se pasa, muy bajo = lento)
    "dropout": 0.25,             # probabilidad de "apagar" neuronas para que no memorice
    "train_path": "./dataset/train",
    "test_path": "./dataset/test",
    "modelo_path": "modelo_melanoma.pth"
}


# ===========================================================================
# TRANSFORMACIONES DE IMÁGENES
# ===========================================================================
# Oliver, antes de meter una foto a la red, hay que prepararla.
# Es como cuando cortas las verduras antes de cocinar.

def get_transforms(es_entrenamiento=False):
    """
    Oliver, aquí preparamos las fotos:
    1. Las hacemos todas del mismo tamaño
    2. Las convertimos a tensores (números que PyTorch entiende)
    3. Normalizamos los colores (para que estén en rangos similares)
    
    Si es entrenamiento, agregamos "trucos" para que la red vea más variedad.
    """
    
    if es_entrenamiento:
        # Data augmentation - Oliver, esto es como hacer trampa legal
        # Le mostramos la misma foto pero volteada, rotada, etc.
        # Así la red aprende mejor sin necesitar más fotos
        return transforms.Compose([
            transforms.Resize((CONFIG["imagen_size"], CONFIG["imagen_size"])),
            transforms.RandomHorizontalFlip(),      # voltea horizontal (50% chance)
            transforms.RandomRotation(15),          # rota hasta 15 grados
            transforms.ColorJitter(brightness=0.2), # varía un poco el brillo
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # Para test/predicción no hacemos augmentation
        # queremos ver el rendimiento real, sin trucos
        return transforms.Compose([
            transforms.Resize((CONFIG["imagen_size"], CONFIG["imagen_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


# ===========================================================================
# LA RED NEURONAL CONVOLUCIONAL (CNN)
# ===========================================================================
# Oliver, aquí es donde se pone bueno. Esta es la arquitectura de tu cerebro artificial.

class MelanomaCNN(nn.Module):
    """
    Oliver, esta clase ES tu modelo de inteligencia artificial.
    
    Imagina que es como un embudo con filtros:
    - Primero pasan las imágenes por capas que detectan patrones simples (bordes, colores)
    - Luego por capas que combinan esos patrones en cosas más complejas (texturas, formas)
    - Al final, todo se aplana y pasa por neuronas que deciden: ¿melanoma o sano?
    """
    
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        
        # ---------------------------------------------------------------
        # BLOQUE 1: Primera capa convolucional
        # ---------------------------------------------------------------
        # Oliver, Conv2d es como pasar un filtro por la imagen
        # (3 canales RGB) -> (32 filtros diferentes)
        # Cada filtro aprende a detectar algo distinto
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # normaliza para que aprenda más estable
        
        # ---------------------------------------------------------------
        # BLOQUE 2: Segunda capa convolucional
        # ---------------------------------------------------------------
        # Oliver, ahora tomamos esos 32 filtros y sacamos 64
        # Es como ir de lo simple a lo complejo
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # ---------------------------------------------------------------
        # BLOQUE 3: Tercera capa convolucional
        # ---------------------------------------------------------------
        # Oliver, una capa más para capturar patrones aún más complejos
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # ---------------------------------------------------------------
        # POOLING Y DROPOUT
        # ---------------------------------------------------------------
        # MaxPool reduce el tamaño a la mitad (como hacer zoom out)
        # Dropout "apaga" neuronas al azar para evitar que memorice
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        
        # ---------------------------------------------------------------
        # CAPAS FULLY CONNECTED (las que deciden)
        # ---------------------------------------------------------------
        # Oliver, aquí calculamos el tamaño después de todas las convoluciones
        # Imagen 128 -> pool -> 64 -> pool -> 32 -> pool -> 16
        tamaño_final = CONFIG["imagen_size"] // 8  # 128/8 = 16
        
        self.fc1 = nn.Linear(128 * tamaño_final * tamaño_final, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 clases: melanoma o sano
    
    def forward(self, x):
        """
        Oliver, este método define el "camino" que sigue la imagen.
        Es como las instrucciones de una receta.
        """
        # Bloque 1: conv -> batchnorm -> relu -> pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Bloque 2: lo mismo
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Bloque 3: lo mismo
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Aplanar - Oliver, convertimos la matriz 3D en un vector 1D
        x = torch.flatten(x, 1)
        
        # Capas fully connected con dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # salida final (sin activación, CrossEntropy la maneja)
        
        return x


# ===========================================================================
# FUNCIONES DE ENTRENAMIENTO
# ===========================================================================

def cargar_datos():
    """
    Oliver, aquí cargamos las fotos de las carpetas.
    ImageFolder es mágico: si tienes carpetas organizadas por clase,
    él solito entiende qué es qué.
    
    Estructura esperada:
    dataset/
        train/
            melanoma/
                foto1.jpg
                foto2.jpg
            sano/
                foto1.jpg
                foto2.jpg
        test/
            (igual)
    """
    print("\n📂 Cargando dataset...")
    
    if not os.path.exists(CONFIG["train_path"]):
        print(f"❌ Oliver, no encuentro la carpeta {CONFIG['train_path']}")
        print("   Crea la estructura: dataset/train/melanoma y dataset/train/sano")
        return None, None, None
    
    train_dataset = ImageFolder(
        root=CONFIG["train_path"], 
        transform=get_transforms(es_entrenamiento=True)
    )
    
    test_dataset = ImageFolder(
        root=CONFIG["test_path"], 
        transform=get_transforms(es_entrenamiento=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True  # mezclar para que no aprenda el orden 
    )
    
    test_loader = DataLoader( 
        test_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False 
    )
    
    clases = train_dataset.classes
    print(f"✅ Clases encontradas: {clases}")
    print(f"   Imágenes de entrenamiento: {len(train_dataset)}")
    print(f"   Imágenes de prueba: {len(test_dataset)}")
    
    return train_loader, test_loader, clases

def entrenar_modelo(): 
    """
    Oliver, aquí es donde la magia ocurre.
    El modelo ve las fotos muchas veces y va ajustando sus "pesos"
    para equivocarse menos. Es como estudiar para un examen.
    """
    print("\n" + "="*50)
    print("🧠 MODO ENTRENAMIENTO")
    print("="*50)
    
    # Cargar datos
    train_loader, test_loader, clases = cargar_datos()
    if train_loader is None:
        return
    
    # Detectar si hay GPU disponible
    # Oliver, si tienes GPU NVIDIA, esto va MUCHO más rápido
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Usando: {device}")
    if device.type == "cuda":
        print(f"   GPU detectada: {torch.cuda.get_device_name(0)}")
    
    # Crear el modelo
    # Oliver, aquí nace tu red neuronal, toda con pesos aleatorios
    print("\n🔨 Creando modelo...")
    modelo = MelanomaCNN().to(device)
    
    # Contar parámetros (para que veas qué tan grande es)
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"   Parámetros totales: {total_params:,}")
    
    # Definir el optimizador y la función de pérdida
    # Oliver, el optimizador es el "profesor" que ajusta los pesos
    # CrossEntropy mide qué tan mal lo está haciendo
    optimizer = optim.Adam(modelo.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Variables para guardar el mejor modelo
    mejor_accuracy = 0
    historico = {"train_loss": [], "test_acc": []}
    
    # ---------------------------------------------------------------
    # LOOP DE ENTRENAMIENTO
    # ---------------------------------------------------------------
    print(f"\n🏋️ Entrenando por {CONFIG['epochs']} épocas...\n")
    
    for epoch in range(CONFIG["epochs"]):
        modelo.train()  # modo entrenamiento (activa dropout)
        total_loss = 0
        correctos = 0
        total = 0
        
        # Oliver, aquí procesamos batch por batch
        for batch_idx, (imagenes, etiquetas) in enumerate(train_loader):
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            
            # 1. Limpiar gradientes anteriores
            optimizer.zero_grad()
            
            # 2. Pasar imágenes por la red (forward pass)
            predicciones = modelo(imagenes)
            
            # 3. Calcular el error
            loss = criterion(predicciones, etiquetas)
            
            # 4. Calcular gradientes (backpropagation)
            # Oliver, esto es la magia: calcula cuánto ajustar cada peso
            loss.backward()
            
            # 5. Actualizar los pesos
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = predicciones.max(1)
            correctos += pred.eq(etiquetas).sum().item()
            total += etiquetas.size(0)
        
        # Calcular métricas de la época
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correctos / total
        
        # Evaluar en test
        test_acc = evaluar_modelo(modelo, test_loader, device)
        
        historico["train_loss"].append(avg_loss)
        historico["test_acc"].append(test_acc)
        
        # Mostrar progreso
        print(f"Época {epoch+1:2d}/{CONFIG['epochs']} │ "
              f"Loss: {avg_loss:.4f} │ "
              f"Train Acc: {train_acc:.1f}% │ "
              f"Test Acc: {test_acc:.1f}%", end="")
        
        # Guardar si es el mejor
        if test_acc > mejor_accuracy:
            mejor_accuracy = test_acc
            guardar_modelo(modelo, clases)
            print(" ✨ Mejor modelo guardado!")
        else:
            print()
    
    print("\n" + "="*50)
    print(f"🎉 Entrenamiento completado!")
    print(f"   Mejor accuracy en test: {mejor_accuracy:.2f}%")
    print(f"   Modelo guardado en: {CONFIG['modelo_path']}")
    print("="*50)


def evaluar_modelo(modelo, loader, device):
    """
    Oliver, aquí probamos qué tan bien está funcionando el modelo.
    Le pasamos fotos que NUNCA ha visto y contamos aciertos.
    """
    modelo.eval()  # modo evaluación (desactiva dropout)
    correctos = 0
    total = 0
    
    with torch.no_grad():  # no calcular gradientes, solo evaluar
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            predicciones = modelo(imagenes)
            _, pred = predicciones.max(1)
            correctos += pred.eq(etiquetas).sum().item()
            total += etiquetas.size(0)
    
    return 100. * correctos / total


def guardar_modelo(modelo, clases):
    """
    Oliver, guardamos todo lo necesario para usar el modelo después:
    - Los pesos de la red
    - Las clases (para saber qué significa cada número)
    - La configuración (por si la necesitas)
    """
    torch.save({
        "model_state_dict": modelo.state_dict(),
        "clases": clases,
        "config": CONFIG
    }, CONFIG["modelo_path"])


# ===========================================================================
# FUNCIONES DE PREDICCIÓN
# ===========================================================================

def cargar_modelo_entrenado():
    """
    Oliver, aquí cargamos el modelo que ya entrenaste.
    """
    if not os.path.exists(CONFIG["modelo_path"]):
        print(f"❌ Oliver, no encuentro el modelo en {CONFIG['modelo_path']}")
        print("   Primero tienes que entrenar con la opción 1")
        return None, None
    
    print(f"\n📦 Cargando modelo desde {CONFIG['modelo_path']}...")
    
    checkpoint = torch.load(CONFIG["modelo_path"], map_location="cpu")
    
    modelo = MelanomaCNN()
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()
    
    clases = checkpoint["clases"]
    print(f"✅ Modelo cargado. Clases: {clases}")
    
    return modelo, clases


def predecir_imagen(ruta_imagen):
    """
    Oliver, aquí clasificamos una imagen nueva.
    La preparamos igual que las de entrenamiento y vemos qué dice la red.
    """
    modelo, clases = cargar_modelo_entrenado()
    if modelo is None:
        return
    
    if not os.path.exists(ruta_imagen):
        print(f"❌ Oliver, no encuentro la imagen: {ruta_imagen}")
        return
    
    print(f"\n🔍 Analizando: {ruta_imagen}")
    
    # Cargar y transformar la imagen
    imagen = Image.open(ruta_imagen).convert("RGB")
    transform = get_transforms(es_entrenamiento=False)
    img_tensor = transform(imagen).unsqueeze(0)  # agregar dimensión de batch
    
    # Hacer predicción
    with torch.no_grad():
        output = modelo(img_tensor)
        probabilidades = F.softmax(output, dim=1)
        confianza, prediccion = torch.max(probabilidades, 1)
    
    clase_predicha = clases[prediccion.item()]
    confianza_pct = confianza.item() * 100
    
    # Mostrar resultados
    print("\n" + "="*40)
    print("📊 RESULTADO")
    print("="*40)
    print(f"   Clasificación: {clase_predicha.upper()}")
    print(f"   Confianza: {confianza_pct:.1f}%")
    print()
    
    # Mostrar probabilidades de cada clase
    print("   Probabilidades:")
    for i, clase in enumerate(clases):
        prob = probabilidades[0][i].item() * 100
        barra = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"   {clase:10s} [{barra}] {prob:.1f}%")
    
    print("="*40)
    
    # Oliver, una advertencia importante
    if clase_predicha == "melanoma":
        print("\n⚠️  IMPORTANTE: Esto es solo una herramienta de apoyo.")
        print("   Consulta siempre con un dermatólogo profesional.")


def predecir_carpeta(ruta_carpeta):
    """
    Oliver, aquí analizamos todas las imágenes de una carpeta de golpe.
    """
    modelo, clases = cargar_modelo_entrenado()
    if modelo is None:
        return
    
    if not os.path.exists(ruta_carpeta):
        print(f"❌ Oliver, no encuentro la carpeta: {ruta_carpeta}")
        return
    
    # Buscar imágenes
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
    imagenes = [f for f in os.listdir(ruta_carpeta) 
                if f.lower().endswith(extensiones)]
    
    if not imagenes:
        print(f"❌ No encontré imágenes en {ruta_carpeta}")
        return
    
    print(f"\n🔍 Analizando {len(imagenes)} imágenes...\n")
    
    transform = get_transforms(es_entrenamiento=False)
    resultados = {"melanoma": 0, "sano": 0}
    
    for nombre in imagenes:
        ruta = os.path.join(ruta_carpeta, nombre)
        imagen = Image.open(ruta).convert("RGB")
        img_tensor = transform(imagen).unsqueeze(0)
        
        with torch.no_grad():
            output = modelo(img_tensor)
            probabilidades = F.softmax(output, dim=1)
            confianza, prediccion = torch.max(probabilidades, 1)
        
        clase = clases[prediccion.item()]
        conf = confianza.item() * 100
        resultados[clase] = resultados.get(clase, 0) + 1
        
        emoji = "🔴" if clase == "melanoma" else "🟢"
        print(f"{emoji} {nombre:30s} -> {clase:10s} ({conf:.1f}%)")
    
    print("\n" + "="*40)
    print("📊 RESUMEN")
    print("="*40)
    for clase, cantidad in resultados.items():
        print(f"   {clase}: {cantidad}")

# ===========================================================================
# MENÚ INTERACTIVO
# ===========================================================================

def mostrar_menu():
    """
    Oliver, el menú principal de tu aplicación.
    """
    print("\n")
    print("╔════════════════════════════════════════════╗")
    print("║   🔬 DETECTOR DE MELANOMA - EDICIÓN OLIVER  ║")
    print("╠════════════════════════════════════════════╣")
    print("║                                            ║")
    print("║   1. 🏋️  Entrenar modelo                   ║")
    print("║   2. 🔍 Predecir una imagen                ║")
    print("║   3. 📁 Predecir carpeta completa          ║")
    print("║   4. ℹ️  Ver configuración actual          ║")
    print("║   5. 🚪 Salir                              ║")
    print("║                                            ║")
    print("╚════════════════════════════════════════════╝")

def ver_configuracion():
    """
    Oliver, aquí puedes ver todos los parámetros actuales.
    """
    print("\n⚙️  CONFIGURACIÓN ACTUAL")
    print("-" * 40)
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print("-" * 40)
    print("\nPara cambiar estos valores, edita el diccionario CONFIG al inicio del archivo.")

def main_menu():
    """
    Oliver, este es el loop principal del menú.
    """
    while True:
        mostrar_menu()
        
        try:
            opcion = input("\n👉 Elige una opción (1-5): ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego Oliver!")
            break
        
        if opcion == "1":
            entrenar_modelo()
        
        elif opcion == "2":
            ruta = input("\n📸 Ruta de la imagen: ").strip()
            if ruta:
                predecir_imagen(ruta)
        
        elif opcion == "3":
            ruta = input("\n📁 Ruta de la carpeta: ").strip()
            if ruta:
                predecir_carpeta(ruta)
        
        elif opcion == "4":
            ver_configuracion()
        
        elif opcion == "5":
            print("\n👋 ¡Hasta luego Oliver! Cuídate ese lunar.")
            break
        
        else:
            print("\n❌ Opción no válida, intenta de nuevo.")
        
        input("\n[Presiona Enter para continuar...]")

# ===========================================================================
# MODO LÍNEA DE COMANDOS
# ===========================================================================
# Oliver, también puedes usar esto desde terminal sin el menú interactivo

def configurar_argumentos():
    """
    Oliver, esto permite usar el programa desde línea de comandos.
    Por ejemplo: python melanoma_detector.py --entrenar
    """
    parser = argparse.ArgumentParser(
        description="🔬 Detector de Melanoma - Edición Oliver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python melanoma_detector.py                     # Abre menú interactivo
  python melanoma_detector.py --entrenar          # Entrena el modelo
  python melanoma_detector.py --predecir foto.jpg # Predice una imagen
  python melanoma_detector.py --carpeta ./fotos   # Predice toda una carpeta
        """
    )
    
    parser.add_argument("--entrenar", action="store_true",
                        help="Entrena el modelo con el dataset")
    parser.add_argument("--predecir", type=str, metavar="IMAGEN",
                        help="Predice una imagen específica")
    parser.add_argument("--carpeta", type=str, metavar="RUTA",
                        help="Predice todas las imágenes de una carpeta")
    parser.add_argument("--epochs", type=int, 
                        help="Número de épocas de entrenamiento")
    
    return parser.parse_args()

# ===========================================================================
# PUNTO DE ENTRADA
# ===========================================================================
# Oliver, aquí empieza todo cuando ejecutas el archivo

if __name__ == "__main__":
    args = configurar_argumentos()
    
    # Si pasaron epochs por línea de comandos, actualizar config
    if args.epochs:
        CONFIG["epochs"] = args.epochs
    
    # Decidir qué hacer según los argumentos
    if args.entrenar:
        entrenar_modelo()
    elif args.predecir:
        predecir_imagen(args.predecir)
    elif args.carpeta:
        predecir_carpeta(args.carpeta)
    else:
        # Sin argumentos = menú interactivo
        main_menu()