import torch
import torch.nn as nn
import torch.nn.functional as F

# Definición del módulo de atención suave (Soft Attention)
class SoftAttention(nn.Module):
    def __init__(self, in_features, attention_size):
        super(SoftAttention, self).__init__()
        self.W_c = nn.Linear(in_features, attention_size)  # Transformación lineal para el query
        self.W_x = nn.Linear(in_features, attention_size)  # Transformación lineal para los elementos de entrada
        self.v = nn.Linear(attention_size, 1)  # Transformación lineal para calcular el peso

    def forward(self, x, query):
        query = self.W_c(query).unsqueeze(1)  # Transformación del query
        x = self.W_x(x)  # Transformación de los elementos de entrada
        s_i = torch.tanh(query + x)  # Cálculo de la puntuación de atención
        alpha_i = F.softmax(self.v(s_i), dim=1)  # Cálculo de los pesos de atención normalizados
        z = (alpha_i * x).sum(dim=1)  # Cálculo del resultado ponderado por atención
        return z

# Definición de la red con atención suave para la detección de cáncer de mama
class BreastCancerDetectionSoftAttentionNet(nn.Module):
    def __init__(self):
        super(BreastCancerDetectionSoftAttentionNet, self).__init__()

        # Definición de la parte de convolución y pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )

        # Capa de Bi-LSTM
        self.blstm = nn.LSTM(128 * 14, 128, num_layers=2, batch_first=True, bidirectional=True)

        # Capa de atención suave (Soft Attention)
        self.soft_attention = SoftAttention(256, 256)

        # Definición de la parte completamente conectada (fully connected)
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),  # Capa lineal
            nn.ReLU(),  # Función de activación ReLU
            nn.Dropout(p=0.5),  # Capa de dropout para regularización
            nn.Linear(in_features=128, out_features=2)  # Capa lineal para las clases
        )

        # Capa de activación Sigmoid para la salida binaria
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Paso de la entrada a través de la parte convolucional y de pooling
        x = self.cnn(x)

        # Reorganización de los datos para la capa BLSTM
        B, C, H, W = x.shape
        x = x.view(B, H, W * C)
        x, _ = self.blstm(x)

        # Último estado de la LSTM como query para la atención suave
        query = x[:, -1, :]
        x = self.soft_attention(x, query)

        # Paso por la parte completamente conectada
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    def __str__(self):
        return "BreastCancerDetectionSoftAttentionNet"