import numpy as np
from torch.optim import RMSprop
import torch
from cos_dist import get_cos_dist_matrix
from metric import NMI, ARI, ACC
from conv import MNISTNetwork


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
            model: [MNISTNetwork],
            epoch: int,
            n_iter: int,
            device: str
    ):
        super().__init__()
        self.model = model().to(device)
        self.cluster = model().to(device)
        self.epoch = epoch
        self.n_iter = n_iter
        self.device = device
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = RMSprop(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.data = None

    def train(self, data):
        self.data = data
        for epoch in range(self.epoch):
            self.model.train()

            for idx, (x, _) in enumerate(self.data):
                if idx == self.n_iter:
                    break

                x = x.view(-1, 1, 28, 28).to(self.device)
                pred = self.model(x)

            cos_dist = get_cos_dist_matrix(pred, self.device)
            loss = self._loss_with_generated_label(cos_dist)

            print(f'Epoch: {epoch} loss: {loss.item()}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._validate()

    def _validate(self):
        self.model.eval()
        y_pred = []
        y_true = []

        for idx, (x, y) in enumerate(self.data):
            x = x.view(-1, 1, 28, 28).to(self.device)
            pred = self.model(x)

            y_pred.append(torch.argmax(pred, 1).detach().cpu().numpy())
            y_true.append(y.numpy())

            if idx == 100:
                break

        pre_y = np.concatenate(y_pred, 0)
        tru_y = np.concatenate(y_true, 0)

        print(f'ACC: {ACC(tru_y, pre_y)}, NMI: {NMI(tru_y, pre_y)}, ARI: {ARI(tru_y, pre_y)}')

    def _loss_with_generated_label(self, cos_dist: torch.tensor):
        generated_label = torch.where(cos_dist >= (1.0 - self.model.eta), 1.0, 0.0).to(self.device)
        return self.criterion(cos_dist, generated_label).to(self.device)
