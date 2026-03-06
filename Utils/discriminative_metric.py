"""""Reimplement TimeGAN-pytorch Codebase in PyTorch.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," NeurIPS 2019.

predictive_metrics.py
Use post-hoc RNN to classify original vs synthetic time series.
Output: discriminative score = abs(acc - 0.5)

Last updated: Auto-generated with PyTorch implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from Utils.metric_utils import train_test_divide, extract_time

class RNNDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNDiscriminator, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)  # h_n: (1, batch, hidden_dim)
        last_hidden = h_n.squeeze(0)        # (batch, hidden_dim)
        logits = self.fc(last_hidden)       # (batch, 1)
        return logits, self.sigmoid(logits)

def batch_generator(data, time, batch_size):
    idx = np.random.permutation(len(data))[:batch_size]
    X_mb = [data[i] for i in idx]
    T_mb = [time[i] for i in idx]
    # Pad sequences to max length in batch
    seqs = [torch.tensor(seq, dtype=torch.float32) for seq in X_mb]
    lengths = torch.tensor(T_mb, dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths

def discriminative_score_metrics(ori_data, generated_data, device=None):
    """Use post-hoc RNN to classify original vs synthetic data"""
    # Default device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data and lengths
    ori_time, ori_max = extract_time(ori_data)
    gen_time, gen_max = extract_time(generated_data)
    max_len = max(ori_max, gen_max)

    # Divide into train/test
    (train_x, train_x_hat, test_x, test_x_hat,
     train_t, train_t_hat, test_t, test_t_hat) = train_test_divide(
        ori_data, generated_data, ori_time, gen_time
    )

    # Convert to tensors
    # Will pad in batch_generator
    input_dim = np.asarray(ori_data).shape[2]
    hidden_dim = input_dim // 2

    # Model, loss, optimizer
    model = RNNDiscriminator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    iterations = 2000
    batch_size = 128

    # Training loop
    model.train()
    for _ in range(iterations):
        # real batch
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        X_mb, T_mb = X_mb.to(device), T_mb.to(device)
        X_hat_mb, T_hat_mb = X_hat_mb.to(device), T_hat_mb.to(device)

        # Labels
        real_labels = torch.ones((X_mb.size(0), 1), device=device)
        fake_labels = torch.zeros((X_hat_mb.size(0), 1), device=device)

        optimizer.zero_grad()
        # Forward real
        logits_real, _ = model(X_mb, T_mb)
        # Forward fake
        logits_fake, _ = model(X_hat_mb, T_hat_mb)
        # Loss
        loss_real = criterion(logits_real, real_labels)
        loss_fake = criterion(logits_fake, fake_labels)
        loss = loss_real + loss_fake
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Test real
        X_test, T_test = batch_generator(test_x, test_t, len(test_x))
        X_test, T_test = X_test.to(device), T_test.to(device)
        logits_real, probs_real = model(X_test, T_test)
        # Test fake
        Xf_test, Tf_test = batch_generator(test_x_hat, test_t_hat, len(test_x_hat))
        Xf_test, Tf_test = Xf_test.to(device), Tf_test.to(device)
        logits_fake, probs_fake = model(Xf_test, Tf_test)

    probs_real = probs_real.cpu().numpy().flatten()
    probs_fake = probs_fake.cpu().numpy().flatten()
    # Combine and compute accuracy
    y_pred = np.concatenate([probs_real > 0.5, probs_fake > 0.5]).astype(int)
    y_true = np.concatenate([np.ones_like(probs_real), np.zeros_like(probs_fake)]).astype(int)
    acc = accuracy_score(y_true, y_pred)
    fake_acc = accuracy_score(np.zeros_like(probs_fake), probs_fake > 0.5)
    real_acc = accuracy_score(np.ones_like(probs_real), probs_real > 0.5)

    discrim_score = abs(acc - 0.5)
    return discrim_score, fake_acc, real_acc
