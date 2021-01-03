"""Implementation of the encoder."""
import torch


class Encoder(torch.nn.Module):
  """The encoder predicts the scores given the text review."""

  def __init__(self, embeddings, hidden_dim, output_dim, drop_out_prob):
    super(Encoder, self).__init__()
    embedding_dim = embeddings.shape[1]
    self.embedding_layer = torch.nn.Embedding.from_pretrained(embeddings)
    self.embedding_layer.weight.requires_grad = True

    self.rnn = torch.nn.RNN(embedding_dim, hidden_dim)
    self.dropout = torch.nn.Dropout(drop_out_prob)
    self.linear = torch.nn.Linear(hidden_dim, output_dim)
    self.activation = torch.nn.Relu()

  def forward(self, inputs):
    embedding = self.embedding_layer(inputs)
    _, hidden = self.rnn(embedding)
    return self.activation(self.linear(hidden))
