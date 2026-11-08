import torch
from torch import nn
 
from cnn import CNN
from rnn import BiLSTM
 
CNN_OUT_CHANNELS = 1024  # число каналов на выходе CNN (последний блок)
 
 
class CRNN(nn.Module):
#  CNN + BiLSTM + Linear.  Вид: (B, 1, 32, W) -> (T, B, num_classes), где T = W // 4, это log-вероятности для CTC loss

    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        self.cnn = CNN()
        self.rnn = BiLSTM(CNN_OUT_CHANNELS, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
 
    def forward(self, x):
        x = self.cnn(x)                    
        x = x.squeeze(2)            
        x = x.permute(0, 2, 1)             
        x = self.rnn(x)                  
        x = self.fc(x)              
        x = torch.log_softmax(x, dim=2)   
        return x.permute(1, 0, 2)        
 
    @staticmethod
    def output_lengths(image_widths):
        return image_widths // 4