import torch
import torch.nn as nn
import torch.nn.functional as F

class HardAttention(nn.Module):
    def __init__(self, in_features, attention_size):
        super(HardAttention, self).__init__()
        # Inicialización de las capas lineales para proyectar la entrada y la consulta
        self.W_c = nn.Linear(in_features, attention_size)
        self.W_x = nn.Linear(in_features, attention_size)
        self.v = nn.Linear(attention_size, 1)

    def forward(self, x, query):
        # Proyectar la consulta usando la capa lineal y aplicar tanh
        query = torch.tanh(self.W_c(query)).unsqueeze(1)
        
        # Proyectar la entrada usando la capa lineal W_x
        x = self.W_x(x)
        
        # Calcular la puntuación de atención
        s_i = torch.tanh(query + x)
        
        # Aplicar softmax a las puntuaciones de atención para obtener los pesos de atención
        alpha_i = F.softmax(self.v(s_i), dim=2)
        
        # Crear un tensor de índices
        indices = torch.arange(x.size(1)).unsqueeze(0)
        
        # Encontrar el índice máximo en las puntuaciones de atención
        selected_idx = torch.argmax(alpha_i, dim=1)
        
        # Utilizar torch.gather para seleccionar elementos de la entrada x según los índices
        z = torch.gather(x, 1, selected_idx.unsqueeze(2)).squeeze(2)
        
        # Devolver el resultado de la atención dura
        return z


# Definición de la red con atención dura para la detección de cáncer de mama
class BreastCancerDetectionHardAttentionNet(nn.Module):
    def __init__(self):
        super(BreastCancerDetectionHardAttentionNet, self).__init__()

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

        # Capa de atención dura (Hard Attention)
        self.hard_attention = HardAttention(256, 256)

        # Definición de la parte completamente conectada (fully connected)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),  # Capa lineal
            nn.ReLU(),  # Función de activación ReLU
            nn.Dropout(p=0.5),  # Capa de dropout para regularización
            nn.Linear(in_features=64, out_features=2)  # Capa lineal para las clases
        )

        # Capa de activación Sigmoid para la salida binaria
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Paso de la entrada a través de la parte convolucional y de pooling
        x = self.cnn(x)

        # Reorganización de los datos para la capa BLSTM
        B, C, H, W = x.shape
        x = x.view(B, H, W * C)
        x, _ = self.blstm(x)

        # Último estado de la LSTM como query para la atención rígida
        query = x[:, -1, :]
        x = self.hard_attention(x, query)

        # Paso por la parte completamente conectada
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    def __str__(self):
        return "BreastCancerDetectionHardAttentionNet"
