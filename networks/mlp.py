from torch import Tensor, nn

class MLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(28 * 28, 64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x: Tensor):
        return self._encoder(x)

class MLPDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

    def forward(self, x: Tensor):
        return self._decoder(x)