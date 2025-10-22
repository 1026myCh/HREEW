
import torch.nn as nn
class AutoEncoder(nn.Module):
    """一维去噪自编码器波形滤波"""
    def __init__(self):
        super().__init__()
        K = 5        # 卷积核心大小，这里设置为5
        S = 2        # 每次计算步长
        P = (K-1)//2 # 补充0长度
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, K, S, padding=P),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, K, S, padding=P),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # 定义解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, K, S, P, output_padding=S-1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 3, K, S, P, output_padding=S-1),
            nn.Tanh() # 约束到-1~1区间，迭代更加稳定
        )
    def forward(self, x):
        h = self.encoder(x) # 编码器构建特征
        y = self.decoder(h) # 解码器输出滤波波形
        return y