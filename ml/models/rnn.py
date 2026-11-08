from torch import nn

class BiLSTM(nn.Module):
#  Двунаправленная обработка.  Вид: (B, 1, 32, W) -> (B, 1024, 1, W // 4)
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x