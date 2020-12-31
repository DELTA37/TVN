import os
import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter


class Solver(object):
    def __init__(self,
                 logs_path,
                 model: nn.Module,
                 train_loader,
                 valid_loader):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optim = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.iteration = 0
        tb_logs = os.path.join(logs_path, 'tb_logs')
        os.makedirs(tb_logs, exist_ok=True)
        self.writer = SummaryWriter(logdir=tb_logs)

        ckpt_path = os.path.join(logs_path, 'ckpt')
        os.makedirs(ckpt_path, exist_ok=True)
        self.ckpt_path = ckpt_path

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, f'model-{self.iteration}.pt'))

    @torch.no_grad()
    def test(self):
        mloss = 0
        mi = 0
        acc = 0
        for x, labels in tqdm(self.valid_loader):
            bs = len(x[1])
            logits = self.model(x)
            acc += torch.sum(torch.eq(torch.argmax(logits, dim=-1), labels).to(dtype=torch.float64)).item()
            loss = self.criterion(logits, labels)
            mloss += loss.item() * bs
            mi += bs
        mloss = mloss / (mi + (mi == 0))
        acc = acc / (mi + (mi == 0))
        self.writer.add_scalar('test/loss', mloss, self.iteration)
        self.writer.add_scalar('test/acc', acc, self.iteration)

    def train(self):
        for x, labels in tqdm(self.train_loader):
            self.optim.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, labels)
            self.writer.add_scalar('train/loss', loss.item(), self.iteration)
            self.writer.add_scalar('train/acc', torch.eq(torch.argmax(logits, dim=-1), labels).mean(), self.iteration)
            loss.backward()
            self.optim.step()
            self.iteration += 1
