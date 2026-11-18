import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN sederhana (2 blok konvolusi) seperti pada M9, dilatih dari nol.

    Perbedaan dari versi M9:
      * ukuran input fleksibel (jumlah fitur FC dihitung otomatis),
      * BatchNorm + Dropout + 1 lapisan tersembunyi (hidden_dim),
      * forward() mengembalikan LOGIT. nn.CrossEntropyLoss sudah memuat log-softmax,
        jadi Softmax TIDAK dipasang di dalam model (menghindari softmax ganda).
        Gunakan torch.softmax(logits, dim=1) saat menghitung probabilitas / ROC-AUC.
    """

    def __init__(self, input_c=3, output=6, img_size=128, hidden_dim=128, dropout=0.5):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        # jumlah fitur setelah flatten dihitung dari ukuran input
        was_training = self.training
        self.eval()
        with torch.no_grad():
            n_features = self.convBlock2(self.convBlock1(torch.zeros(1, input_c, img_size, img_size))).numel()
        self.train(was_training)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=n_features, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output),
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return self.fc(torch.flatten(x, 1))
