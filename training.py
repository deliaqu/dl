"""Training loop for the rationale model."""
import gzip
import json

from absl import app
from absl import flags
import encoder
import generator
import torch
from torch import F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

FLAGS = flags.FLAGS
FLAGS.DEFINE_integer('epochs', 10, 'Number of epochs to train the model for.')
FLAGS.DEFINE_float('learning_rate', 0.001,
                   'Learning rate used to train the model.')
FLAGS.DEFINE_string('embedding', '', 'Path to the word embedding vectors.')
FLAGS.DEFINE_string('train_data', '', 'Path to the training set')
FLAGS.DEFINE_string('dev_data', '', 'Path to the development set')
FLAGS.DEFINE_string('rationales', '', 'Rationale annotations used for testing.')
FLAGS.DEFINE_integer(
    'aspect', '-1', 'Which aspect to predict the score for.'
    '-1 means all aspects.')
FLAGS.DEFINE_string('output', '', 'Path to output the selected rationales and'
                    'predictions.')
FLAGS.DEFINE_float('lambda_selection_cost', 0.0003,
                   'Regularization factor for the selection cost.')
FLAGS.DEFINE_float('lambda_continuity_cost', 2.0,
                   'Regularization factor for the continuity cost.')
FLAGS.DEFINE_float('drop_out_prob_encoder', 0.3, 'Dropout to be applied to'
                   'the encoder.')
FLAGS.DEFINE_float('drop_out_prob_generator', 0.3, 'Dropout to be applied to'
                   'the generator.')
FLAGS.DEFINE_integer('hidden_dim_encoder', 256, 'Hidden dimension for the'
                     'encoder RNN.')
FLAGS.DEFINE_integer('hidden_dim_generator', 256, 'Hidden dimension for the'
                     'generator RNN.')
flags.mark_flag_as_required('embedding')
flags.mark_flag_as_required('train_data')
flags.mark_flag_as_required('dev_data')


def read_rationales(path):
  """Utility function to read the rationales from path."""
  data = []
  fopen = gzip.open if path.endswith('.gz') else open
  with fopen(path) as fin:
    for line in fin:
      item = json.loads(line)
      data.append(item)
  return data


def read_data(path):
  """Utility function to read data from path."""
  data_x, data_y = [], []
  fopen = gzip.open if path.endswith('.gz') else open
  with fopen(path) as fin:
    for line in fin:
      y, _, x = line.partition('\t')
      x, y = x.split(), y.split()
      if not x:
        continue
      y = torch.tensor([float(v) for v in y], dtype=torch.float64)
      data_x.append(x)
      data_y.append(y)
  return data_x, data_y


class BeersReviewDataSet(Dataset):
  """Used to load data."""

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    return self.x[index], self.y[index]


def read_embedding(path):
  """Reads pretrained embeddings from path."""
  lines = []
  with gzip.open(path) as file:
    lines = file.readlines()
  embedding_tensors = []
  word_to_indx = {}
  for indx, l in enumerate(lines):
    word, emb = l.split()[0], l.split()[1:]
    vector = [float(x) for x in emb]
    if indx == 0:
      embedding_tensors.append(torch.zeros(len(vector)))
    embedding_tensors.append(vector)
    word_to_indx[word] = indx + 1
  embedding = torch.tensor(embedding_tensors, dtype=torch.float32)
  return embedding, word_to_indx


def main(argv):
  del argv
  x_train, y_train = read_data(FLAGS.train_data)
  loader = iter(
      DataLoader(
          BeersReviewDataSet(x_train, y_train), batch_size=32, shuffle=True))
  embedding = read_embedding(FLAGS.embedding)
  enc = encoder.Encoder(embedding, FLAGS.hidden_dim_encoder, len(y_train[0]),
                        FLAGS.drop_out_prob_encoder)
  gen = generator.Generator(embedding, FLAGS.hidden_dim_generator,
                            FLAGS.drop_out_prob_generator)
  optimizer = torch.optim.Adam([enc.parameters(), gen.parameters()],
                               lr=FLAGS.learning_rate)
  for i in range(FLAGS.epochs):
    print('-------------\nEpoch {}:\n'.format(i))
    losses = []
    for batch, labels in loader:
      optimizer.zero_grad()
      selection = gen.select(gen(batch))
      selection_cost, continuity_cost = gen.loss(selection, batch)
      batch = batch * selection.unsqueeze(-1)
      logit = enc(batch)
      loss = F.mse_loss(logit, labels.float())
      loss += FLAGS.lambda_selection_cost * selection_cost
      loss += FLAGS.lambda_continuity_cost * continuity_cost
      loss.backward()
      losses.append(loss)
      optimizer.step()
    print('Loss: ', torch.mean(losses))


if __name__ == '__main__':
  app.run(main)
