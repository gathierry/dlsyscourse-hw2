import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    blocks = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
    ]
    for _ in range(num_blocks):
        blocks.append(
            ResidualBlock(
                hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob
            )
        )
    blocks.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*blocks)


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is not None:
        model.train()
    else:
        model.eval()
    loss_func = nn.SoftmaxLoss()
    sum_loss = 0
    sum_err = 0
    cnt = 0
    for data in dataloader:
        X, y = data
        print(X.shape)
        y_hat = model(X)
        loss = loss_func(y_hat, y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        sum_loss += loss.numpy() * X.shape[0]
        y_hat = np.argmax(y_hat.numpy(), axis=1)
        sum_err += (y_hat != y.numpy()).sum()
        cnt += X.shape[0]
    return sum_err / cnt, sum_loss / cnt
        

def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        test_err, test_loss = epoch(test_dataloader, model)
    return train_err, train_loss, test_err, test_loss



if __name__ == "__main__":
    train_mnist(data_dir="../data")
