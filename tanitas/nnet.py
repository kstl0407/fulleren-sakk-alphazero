import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "alpha-zero-general-master"))
from NeuralNet import NeuralNet

from game import FullerenSakkGame, NUM_CHANNELS, NUM_TILES, ACTION_SIZE

INPUT_DIM = NUM_CHANNELS * NUM_TILES

DEFAULT_ARGS = {
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "dropout":        0.3,
    "epochs":         10,
    "batch_size":     64,
    "cuda":           torch.cuda.is_available(),
    "hidden_dim":     512,
    "num_resblocks":  6,
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "checkpoints"),
}


class ResBlock(nn.Module):
    """Fully-connected residual block with BatchNorm and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two linear layers with a residual connection."""
        res = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.bn2(self.fc2(x))
        return F.relu(x + res)


class FullerenNet(nn.Module):
    """Policy-value network: flattened board state → (action logits, scalar value)."""

    def __init__(self, args: dict = None):
        super().__init__()
        if args is None:
            args = DEFAULT_ARGS
        h    = args["hidden_dim"]
        drop = args["dropout"]

        self.input_fc = nn.Linear(INPUT_DIM, h)
        self.input_bn = nn.BatchNorm1d(h)

        self.res_blocks = nn.Sequential(
            *[ResBlock(h, drop) for _ in range(args["num_resblocks"])]
        )

        self.policy_fc1  = nn.Linear(h, 512)
        self.policy_drop = nn.Dropout(drop)
        self.policy_fc2  = nn.Linear(512, ACTION_SIZE)

        self.value_fc1   = nn.Linear(h, 256)
        self.value_drop  = nn.Dropout(drop)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """Return raw policy logits and a tanh-bounded scalar value."""
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_bn(self.input_fc(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_fc1(x))
        p = self.policy_drop(p)
        p = self.policy_fc2(p)

        v = F.relu(self.value_fc1(x))
        v = self.value_drop(v)
        v = torch.tanh(self.value_fc2(v))

        return p, v


class FullerenNNet(NeuralNet):
    """AlphaZero NeuralNet wrapper around FullerenNet."""

    def __init__(self, game: FullerenSakkGame, args: dict = None):
        self.game   = game
        self.args   = args or DEFAULT_ARGS
        self.nnet   = FullerenNet(self.args)
        self.device = torch.device("cuda" if self.args["cuda"] else "cpu")
        self.nnet.to(self.device)
        os.makedirs(self.args["checkpoint_dir"], exist_ok=True)

    def train(self, examples):
        """Train on a list of (board, pi, v) examples for self.args['epochs'] epochs."""
        optimizer = optim.Adam(
            self.nnet.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5
        )

        boards = torch.FloatTensor(np.array([e[0] for e in examples])).to(self.device)
        pis    = torch.FloatTensor(np.array([e[1] for e in examples])).to(self.device)
        vs     = torch.FloatTensor(np.array([e[2] for e in examples])).unsqueeze(1).to(self.device)

        loader = DataLoader(
            TensorDataset(boards, pis, vs),
            batch_size=self.args["batch_size"],
            shuffle=True,
            drop_last=len(examples) > self.args["batch_size"],
        )

        n = len(loader)
        self.nnet.train()
        for epoch in range(self.args["epochs"]):
            total_pi = total_v = 0.0
            for b_boards, b_pi, b_v in loader:
                raw_p, v_out = self.nnet(b_boards)

                # Masked cross-entropy: MCTS policy only has mass on valid moves,
                # but network logits are unmasked — log_softmax over all actions.
                log_pi  = F.log_softmax(raw_p, dim=1)
                pi_loss = -torch.sum(b_pi * log_pi, dim=1).mean()
                v_loss  = F.mse_loss(v_out, b_v)
                loss    = pi_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_pi += pi_loss.item()
                total_v  += v_loss.item()

            scheduler.step((total_pi + total_v) / n)
            print(f"  Epoch {epoch+1:02d}/{self.args['epochs']} | "
                  f"pi={total_pi/n:.4f} | v={total_v/n:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
        # v2: az utolsó epoch átlagvesztesége (a Coach az iteraciok.csv-be írja)
        return (total_pi / n, total_v / n) if n else None

    def predict(self, board: np.ndarray):
        """Return (softmax policy, scalar value) for a single board."""
        t = torch.FloatTensor(board).unsqueeze(0).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            raw_p, v = self.nnet(t)
        pi = torch.softmax(raw_p, dim=1).squeeze(0).cpu().numpy()
        return pi, v.item()

    def save_checkpoint(self, folder: str = None, filename: str = "checkpoint.pth.tar"):
        """Save network weights and args to a .pth.tar file."""
        folder = folder or self.args["checkpoint_dir"]
        os.makedirs(folder, exist_ok=True)
        torch.save(
            {"state_dict": self.nnet.state_dict(), "args": self.args},
            os.path.join(folder, filename),
        )

    def load_checkpoint(self, folder: str = None, filename: str = "checkpoint.pth.tar"):
        """Load network weights from a .pth.tar checkpoint file."""
        folder = folder or self.args["checkpoint_dir"]
        path   = os.path.join(folder, filename)
        ckpt   = torch.load(path, map_location=self.device)
        self.nnet.load_state_dict(ckpt["state_dict"])
        print(f"Checkpoint betöltve: {path}")


if __name__ == "__main__":
    g    = FullerenSakkGame()
    nnet = FullerenNNet(g)
    board = g.getInitBoard()

    pi, v = nnet.predict(board)
    print(f"Policy alak:  {pi.shape}")
    print(f"Policy összeg:{pi.sum():.6f}")
    print(f"Value:        {v:.4f}")

    total = sum(p.numel() for p in nnet.nnet.parameters() if p.requires_grad)
    print(f"Paraméterek:  {total:,}")
