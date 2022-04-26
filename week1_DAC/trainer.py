import numpy as np
from torch.nn import CrossEntropyLoss, BCELoss
from torch import nn
from torch.optim import SGD, RMSprop, Adam
import torch
from cos_dist import get_cos_dist_matrix
from metric import NMI, ARI, ACC


def check_shape(x):
    import torch
    import numpy as np
    import tensorflow as tf

    if torch.is_tensor(x):
        print(x.size())
    elif isinstance(x, np.ndarray):
        print(x.shape)
    elif isinstance(x, type([])):
        print(np.array(x).shape)
    elif tf.is_tensor(x):
        print(tf.shape(x))


class Trainer(object):
    def __init__(
            self,
            model: nn.Module,
            epoch: int,
            n_iter: int,
            device: str
    ):
        super().__init__()
        self.model = model().to(device)
        self.cluster = model().to(device)
        self.epoch = epoch
        self.n_iter = n_iter
        # self.upper_thr = 0.99
        # self.lower_thr = 0.75
        # self.eta = (self.upper_thr - self.lower_thr) / self.epoch
        self.device = device
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = RMSprop(self.model.parameters(), lr=1e-3, momentum=0.9)

    def train(self, data):
        for epoch in range(self.epoch):
            print(self.model.thr)
            self.model.train()

            for idx, (x, _) in enumerate(data):
                if idx == self.n_iter:
                    break

                x = x.view(-1, 1, 28, 28).to(self.device)
                pred = self.model(x)

            cos_dist = get_cos_dist_matrix(pred, self.device)
            # print(cos_dist)
            loss = self._loss_with_generated_label(cos_dist)

            print(f'Epoch: {epoch}, Iteration:{idx}, loss: {loss.item()}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            y_pred = []
            y_true = []

            for idx, (x, y) in enumerate(data):
                x = x.view(-1, 1, 28, 28).to(self.device)
                pred = self.model(x)

                y_pred.append(torch.argmax(pred, 1).detach().cpu().numpy())
                y_true.append(y.numpy())

                if idx == 100:
                    break

            pre_y = np.concatenate(y_pred, 0)
            tru_y = np.concatenate(y_true, 0)

            print(f'ACC: {ACC(tru_y, pre_y)}, NMI: {NMI(tru_y, pre_y)}, ARI: {ARI(tru_y, pre_y)}')

            # self.thr -= (0.9999 - 0.5) / self.epoch
            # self.upper_thr -= self.eta
            # self.lower_thr += self.eta

    def _loss_with_generated_label(self, cos_dist: torch.tensor):
        r_label = torch.where(cos_dist >= self.model.thr, 1.0, 0.0).to(self.device)
        # r_u = torch.where(cos_dist >= self.upper_thr, 1.0, 0.0).to(self.device)
        # r_l = torch.where(cos_dist < self.lower_thr, 1.0, 0.0).to(self.device)
        # r_label = torch.where(
        #     cos_dist >= (self.upper_thr + self.lower_thr) / 2, 1.0, 0.0
        # ).to(self.device)
        # loss = torch.sum(
        #     (r_u + r_l) * self.criterion(cos_dist, r_label)
        # ) / torch.sum(r_u + r_l).to(self.device)

        # r_label = torch.where(cos_dist >= self.upper_thr, 1.0, 0.0)
        loss = self.criterion(cos_dist, r_label).to(self.device)
        return loss
