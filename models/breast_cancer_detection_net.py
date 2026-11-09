import torch.nn as nn

class BreastCancerDetectionNet(nn.Module):
    def __init__(self):
        super(BreastCancerDetectionNet, self).__init__()

        # Definición de la parte de convolución y pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # Capa convolucional
            nn.BatchNorm2d(64),  # Capa de normalización por lotes
            nn.ReLU(),  # Función de activación ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Capa de pooling
            nn.Dropout(p=0.25),  # Capa de dropout para regularización

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # Capa convolucional
            nn.BatchNorm2d(128),  # Capa de normalización por lotes
            nn.ReLU(),  # Función de activación ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Capa de pooling
            nn.Dropout(p=0.25)  # Capa de dropout para regularización
        )

        # Definición de la parte completamente conectada (fully connected)
        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 14 * 14, out_features=128),  # Capa lineal
            nn.ReLU(),  # Función de activación ReLU
            nn.Dropout(p=0.5),  # Capa de dropout para regularización
            nn.Linear(in_features=128, out_features=2)  # Capa lineal para las clases
        )

        # Capa de activación Sigmoid para la salida binaria
        #self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Paso de la entrada a través de la parte convolucional y de pooling
        x = self.cnn(x)

        # Aplanamiento para la parte completamente conectada
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Aplicación de la función de activación Sigmoid a la salida
        x = self.sigmoid(x)

        return x

    def __str__(self):
        return "BreastCancerDetectionNet"