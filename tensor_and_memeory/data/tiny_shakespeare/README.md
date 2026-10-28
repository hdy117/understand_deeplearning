# Tiny Shakespeare

This directory contains a local copy of the public Tiny Shakespeare text from
Andrej Karpathy's `char-rnn` repository. The raw file is about 1.1 MB, so it is
small enough for a teaching demo and does not require a dataset library.

The dataset uses character-level tokenization. Every character gets one integer
ID. A language-model sample is a pair of overlapping windows:

```text
x: To be, or not to be
y: o be, or not to be\n
```

The files are:

- `input.txt`: original text;
- `train.pt`, `valid.pt`, `test.pt`: one-dimensional `torch.long` token streams;
- `metadata.json`: vocabulary, split sizes, and source URL;
- `prepare.py`: reproducible preprocessing script;
- `dataset.py`: shared `Dataset` for RNN, LSTM, and Transformer demos.

Prepare or rebuild the derived files with:

```bash
python data/tiny_shakespeare/prepare.py
```

Minimal loader:

```python
from torch.utils.data import DataLoader
from data.tiny_shakespeare.dataset import CharLanguageModelDataset

dataset = CharLanguageModelDataset(split="train", seq_len=128)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
x, y = next(iter(loader))  # x, y: [batch, sequence]
```

Embed `x` before feeding it to a model. For each architecture, the same target
`y` works with causal cross-entropy over the vocabulary.
