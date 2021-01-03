"""Implementation of the encoder."""
import torch


class Generator(torch.nn.Module):
  """The generator selects the parts of the text review used for the encoder."""

  def __init__(self, embeddings, hidden_dim, drop_out_prob):
    super(Generator, self).__init__()
    vocab_size, embedding_dim = embeddings.shape
    self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    self.embedding_layer.weight.data = torch.from_numpy(embeddings)
    self.embedding_layer.weight.requires_grad = False

    self.rnn = torch.nn.RNN(embedding_dim, hidden_dim, bidirectional=True)
    self.dropout = torch.nn.Dropout(drop_out_prob)
    self.linear = torch.nn.Linear(2 * hidden_dim, 1)
    self.activation = torch.nn.Relu()

  def forward(self, inputs):
    embedding = self.embedding_layer(inputs)
    out, _ = self.rnn(embedding)
    return self.activation(self.linear(self.dropout(out)))

  def select(self, logits):
    out = (logits > 0).float()
    return out

  def loss(self, selection, inputs):
    selection_cost = torch.mean(torch.sum(selection, dim=1))
    l_padded_mask = torch.cat([selection[:, 0].unsqueeze(1), selection], dim=1)
    r_padded_mask = torch.cat([selection, selection[:, -1].unsqueeze(1)], dim=1)
    continuity_cost = torch.mean(
        torch.sum(torch.abs(l_padded_mask - r_padded_mask), dim=1))
    return selection_cost, continuity_cost
