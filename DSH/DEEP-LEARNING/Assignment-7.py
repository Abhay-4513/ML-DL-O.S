import torch
import torch.nn as nn
class CIFAR_FNN(nn.Module):
    def __init__(self, input_dim=32*32*3, hidden_dim=512, num_classes=10):
        super(CIFAR_FNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    class IMDB_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=1):
        super(IMDB_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)
    if __name__ == "__main__":
    # Test FNN
    fnn = CIFAR_FNN()
    dummy_image = torch.randn(4, 3, 32, 32)
    print("FNN Output Shape:", fnn(dummy_image).shape)

    # Test CNN
    cnn = CIFAR_CNN()
    print("CNN Output Shape:", cnn(dummy_image).shape)

    # Test LSTM
    lstm = IMDB_LSTM(vocab_size=10000)
    dummy_text = torch.randint(0, 10000, (4, 256)) # Batch of 4, sequence length 256
    print("LSTM Output Shape:", lstm(dummy_text).shape)
