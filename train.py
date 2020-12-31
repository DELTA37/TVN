import os
import argparse
from torch.utils.data import Dataset, DataLoader
from tvn.solver import Solver
from tvn.model import TVN, no_verbose
from tvn.config import CFG1
from tvn.data import SomethingSomethingV2
from torch.multiprocessing import cpu_count


if __name__ == '__main__':
    no_verbose()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', default='./data', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--logs_path', default='./logs', type=str)
    parser.add_argument('--num_epochs', default=100, type=int)
    args = parser.parse_args()

    train_dataset = SomethingSomethingV2(root=args.data_root, mode='train')
    valid_dataset = SomethingSomethingV2(root=args.data_root, mode='validation')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=cpu_count())

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=cpu_count())

    tvm = TVN(CFG1, num_classes=len(train_dataset.classes))

    solver = Solver(args.logs_path,
                    model=tvm,
                    train_loader=train_loader,
                    valid_loader=valid_loader)

    solver.test()
    for _ in range(args.num_epochs):
        solver.train()
        solver.test()
        solver.save()
