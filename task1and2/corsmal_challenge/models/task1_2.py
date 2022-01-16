import torch
import torch.nn.functional as F
from torch import nn

from corsmal_challenge.models.audio import LogMelEncoder


class T1Head(nn.Module):
    def __init__(self, embed_dim: int = 128, classify_num: int = 3):
        super(T1Head, self).__init__()
        self.classify_num = classify_num
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, classify_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class T2Head(nn.Module):
    def __init__(self, embed_dim: int = 128, classify_num: int = 4):
        super(T2Head, self).__init__()
        self.classify_num = classify_num
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, classify_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class T1HeadV2(nn.Module):
    def __init__(self, embed_dim: int = 128, classify_num: int = 3):
        super(T1HeadV2, self).__init__()
        self.classify_num = classify_num
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, classify_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class TaskChallenger(nn.Module):
    def __init__(self, task_id: int = 1):
        super(TaskChallenger, self).__init__()
        self.task_id = task_id
        self.encoder = LogMelEncoder(num_encoder_blocks=6, num_heads=8)
        self.classify_head1 = T1Head()
        self.classify_head2 = T2Head()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.encoder(inputs)
        if self.task_id == 1:
            x = self.classify_head1(x[:, 0, :])  # extract embedding of class token
        elif self.task_id == 2:
            x = self.classify_head2(x[:, 0, :])  # extract embedding of class token
        x = x.squeeze(1)
        return x


class TaskChallenger2(nn.Module):
    def __init__(self, task_id: int = 1):
        super(TaskChallenger2, self).__init__()
        self.task_id = task_id
        self.encoder = LogMelEncoder(num_encoder_blocks=4, num_heads=4)
        self.classify_head1 = T1Head()
        self.classify_head2 = T2Head()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.encoder(inputs)
        if self.task_id == 1:
            x = self.classify_head1(x[:, 0, :])  # extract embedding of class token
        elif self.task_id == 2:
            x = self.classify_head2(x[:, 0, :])  # extract embedding of class token
        x = x.squeeze(1)
        return x
