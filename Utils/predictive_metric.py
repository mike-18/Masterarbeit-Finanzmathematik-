"""
Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from Utils.metric_utils import extract_time


class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        y_hat_logit = self.fc(out)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    """
    Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no, seq_len, dim = ori_data.shape
    ori_time, ori_max_seq_len = extract_time(ori_data)
    gen_time, gen_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, gen_max_seq_len)

    hidden_dim = dim // 2
    iterations = 5000
    batch_size = 128

    model = GRUPredictor(input_dim=dim-1, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.L1Loss()

    for _ in range(iterations):
        idx = np.random.permutation(len(generated_data))[:batch_size]
        X_mb = [torch.tensor(generated_data[i][:-1, :(dim - 1)], dtype=torch.float32) for i in idx]
        T_mb = [gen_time[i] - 1 for i in idx]
        Y_mb = [torch.tensor(generated_data[i][1:, (dim - 1)].reshape(-1, 1), dtype=torch.float32) for i in idx]

        lengths = torch.tensor(T_mb, dtype=torch.long).to(device)
        X_pad = nn.utils.rnn.pad_sequence(X_mb, batch_first=True).to(device)
        Y_pad = nn.utils.rnn.pad_sequence(Y_mb, batch_first=True).to(device)

        model.train()
        optimizer.zero_grad()
        Y_pred = model(X_pad, lengths)
        loss = loss_fn(Y_pred, Y_pad)
        loss.backward()
        optimizer.step()

    # Evaluate on original data
    idx = np.random.permutation(len(ori_data))[:no]
    X_mb = [torch.tensor(ori_data[i][:-1, :(dim - 1)], dtype=torch.float32) for i in idx]
    T_mb = [ori_time[i] - 1 for i in idx]
    Y_mb = [ori_data[i][1:, (dim - 1)].reshape(-1, 1) for i in idx]

    lengths = torch.tensor(T_mb, dtype=torch.long).to(device)
    X_pad = nn.utils.rnn.pad_sequence(X_mb, batch_first=True).to(device)

    model.eval()
    with torch.no_grad():
        Y_pred = model(X_pad, lengths).cpu().numpy()

    mae = 0
    for i in range(no):
        mae += mean_absolute_error(Y_mb[i], Y_pred[i, :len(Y_mb[i]), :])
    predictive_score = mae / no

    return predictive_score
