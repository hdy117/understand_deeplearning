import torch
import torch.nn as nn
import argparse


class RNNCell(nn.Module):
    def __init__(self, d_model_in: int = 64, d_model_out: int = 64):
        super().__init__()

        self.current_fn = nn.Linear(in_features=d_model_in, out_features=d_model_out)
        self.memory_fn = nn.Linear(in_features=d_model_out, out_features=d_model_out)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        # x:[batch, d_model_in]
        # h:[batch, d_model_out], h is the memory of previous RNN cell
        # out:[batch, d_model_out]
        memory = self.memory_fn(h)
        new_info = self.current_fn(x)
        cur_activate = torch.tanh(memory + new_info)
        return cur_activate


class ToyRNN(nn.Module):
    def __init__(
        self,
        n_layer: int = 3,
        embedding_size: int = 8,
        d_model: int = 64,
        output_size: int = 65,
    ):
        super().__init__()
        self.n_layer: int = n_layer
        self.d_model: int = d_model
        self.embedding_size: int = embedding_size
        self.output_size: int = output_size

        self.layer_list = nn.ModuleList(
            [RNNCell(d_model_in=embedding_size, d_model_out=d_model)]
            + [
                RNNCell(d_model_in=d_model, d_model_out=d_model)
                for _ in range(n_layer - 1)
            ]
        )
        self.embedding_layer = nn.Embedding(
            num_embeddings=output_size, embedding_dim=embedding_size
        )
        self.lm_head = nn.Linear(in_features=d_model, out_features=output_size)

    def _print_meta_(self):
        print(
            f"self.n_layer:{self.n_layer}, self.d_model:{self.d_model}, self.input_size:{self.input_size}"
        )

    def forward(self, x):
        # x: [batch, tokens]
        embed = self.embedding_layer(x)
        B, S, _ = embed.shape  # [batch, sequence, embedding_size]
        layer_out = embed
        h_out = []
        for layer in self.layer_list:
            h = torch.zeros(size=(B, self.d_model))
            outs = []
            for seq_i in S:
                h = layer.forward(layer_out[:, seq_i], h)
                outs.append(h)
            h_out.append(h)  # [batch, d_model], only the last h of an RNNCell
            layer_out = torch.stack(
                outs, dim=1
            )  # [batch, S, d_model], all h while a RNNCell loop S times
        # adjust for text generation task
        logits = self.lm_head(layer_out)
        return logits, torch.stack(h_out, dim=0)


from torch.utils.data import DataLoader
from data.tiny_shakespeare.dataset import CharLanguageModelDataset
import json
from pathlib import Path

meta_path = Path(__file__).resolve().parent / "data/tiny_shakespeare/metadata.json"


class Config:
    lr: float = 1e-3
    epoch: int = 1000
    batch_size: int = 100
    seq_len: int = 16
    cosine_anneal_T_0: int = epoch / batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    itos: list[str] = []

    @classmethod
    def get_itos_dict(cls) -> dict:
        json_dict = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"json_dict:{json_dict}")
        assert "itos" in json_dict.keys()
        return json_dict["itos"]

    @classmethod
    def int_list_to_str(cls, int_list: list[int]) -> str:
        out_str: str = ""
        for i in int_list:
            out_str = out_str + Config.itos[i]
        return out_str


Config.itos = Config.get_itos_dict()


def dataset_test():
    dataset = CharLanguageModelDataset(split="train", seq_len=128)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    x, y = next(iter(loader))

    assert torch.equal(x[:, 1:], y[:, :-1]), "targets must be shifted by one character"
    print(f"vocab_size: {len(Config.itos)}, x: {tuple(x.shape)}, y: {tuple(y.shape)}")
    for row in range(x.size(0)):
        x_text = Config.int_list_to_str(x[row].tolist())
        y_text = Config.int_list_to_str(y[row].tolist())
        print(f"sample {row} x: {repr(x_text[:30])}")
        print(f"sample {row} y: {repr(y_text[:30])}")
        print(f"shift aligned: {x_text[1:] == y_text[:-1]}")


def train(rnn: ToyRNN = None):
    toy_rnn: ToyRNN = ToyRNN(
        n_layer=4, embedding_size=8, d_model=128, output_size=len(Config.itos)
    ).to(Config.device)

    optimizer = torch.optim.AdamW(
        toy_rnn.parameters(),
        lr=Config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.cosine_anneal_T_0
    )

    loss_fn = torch.nn.functional.cross_entropy()

    # data set
    dataset = CharLanguageModelDataset(split="train", seq_len=Config.seq_len)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    print(
        f"seq_len:{Config.seq_len}, batch_size:{Config.batch_size}, device:{Config.device}"
    )

    # train
    for epoch_i in range(Config.epoch):
        for index, (x, y) in enumerate(dataloader):
            x, y_label = x.to(Config.device), y.to(Config.device)
            y_predict, h = toy_rnn.forward(x)

            loss = loss_fn(y_predict.transpose(1, 2), y_label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def eval(rnn: ToyRNN = None):
    pass


def set_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="parser for RNN")
    parser.add_argument(
        "--run_mode", type=str, default="train", help=" train or eval, default is train"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args: argparse.Namespace = set_args()

    rnn: ToyRNN = ToyRNN(n_layer=3, d_model=64)
    rnn._print_meta_()

    dataset_test()

    if args.run_mode == "train":
        print(f"run in {args.run_mode} mode")
    else:
        print(f"run in {args.run_mode} mode")
