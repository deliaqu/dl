"""Training loop for the rationale model."""
import gzip
import json

from absl import app
from absl import flags
import encoder
import generator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train the model for.')
flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate used to train the model.')
flags.DEFINE_string('embedding', '', 'Path to the word embedding vectors.')
flags.DEFINE_string('train_data', '', 'Path to the training set')
flags.DEFINE_string('dev_data', '', 'Path to the development set')
flags.DEFINE_string('rationales', '', 'Rationale annotations used for testing.')
flags.DEFINE_integer(
    'aspect', '-1', 'Which aspect to predict the score for.'
    '-1 means all aspects.')
flags.DEFINE_string('output', '', 'Path to output the selected rationales and'
                    'predictions.')
flags.DEFINE_float('lambda_selection_cost', 0.0003,
                   'Regularization factor for the selection cost.')
flags.DEFINE_float('lambda_continuity_cost', 2.0,
                   'Regularization factor for the continuity cost.')
flags.DEFINE_float('drop_out_prob_encoder', 0.3, 'Dropout to be applied to'
                   'the encoder.')
flags.DEFINE_float('drop_out_prob_generator', 0.3, 'Dropout to be applied to'
                   'the generator.')
flags.DEFINE_integer('hidden_dim_encoder', 256, 'Hidden dimension for the'
                     'encoder RNN.')
flags.DEFINE_integer('hidden_dim_generator', 256, 'Hidden dimension for the'
                     'generator RNN.')
PADDING = '<pad>'


def read_rationales(path):
  """Utility function to read the rationales from path."""
  data = []
  fopen = gzip.open if path.endswith('.gz') else open
  with fopen(path) as fin:
    for line in fin:
      item = json.loads(line)
      data.append(item)
  return data


def process_data(data, nil_idx):
  maxlen = max(len(x) for x in data)
  results = []
  for x in data:
    if len(x) < maxlen:
      x.extend([nil_idx for i in range(maxlen - len(x))])
    results.append(torch.tensor([v for v in x], dtype=torch.int64))
  return results


def read_data(path, word_to_idx):
  """Utility function to read data from path."""
  data_x, data_y = [], []
  with open(path, 'r') as f:
    for line in f:
      y, _, x = line.partition('\t')
      x, y = x.split(), y.split()
      if not x:
        continue
      x_int = []
      for word in x:
        if word in word_to_idx:
          x_int.append(word_to_idx[word])
        else:
          x_int.append(len(word_to_idx))
      data_x.append(x_int)
      y = torch.tensor([float(v) for v in y], dtype=torch.float64)
      data_y.append(y)
  return process_data(data_x, len(word_to_idx)), data_y


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
  with open(path, 'r') as file:
    lines = file.readlines()
  embedding_tensors = []
  word_to_indx = {}
  for indx, l in enumerate(lines):
    word, emb = l.split()[0], l.split()[1:]
    vector = [float(x) for x in emb]
    if len(vector) != 200:
      continue
    if indx == 0:
      embedding_tensors.append([0 for i in range(len(vector))])
    embedding_tensors.append(vector)
    word_to_indx[word] = indx + 1
  embeddings = torch.tensor(embedding_tensors, dtype=torch.float32)
  return embeddings, word_to_indx


def main(argv):
  del argv
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  embeddings, word_to_indx = read_embedding(FLAGS.embedding)
  x_train, y_train = read_data(FLAGS.train_data, word_to_indx)
  x_dev, y_dev = read_data(FLAGS.dev_data, word_to_indx)
  loader = DataLoader(
      BeersReviewDataSet(x_train, y_train), batch_size=32, shuffle=True)
  enc = encoder.Encoder(embeddings, FLAGS.hidden_dim_encoder, len(y_train[0]),
                        FLAGS.drop_out_prob_encoder)
  enc.to(device)
  gen = generator.Generator(embeddings, FLAGS.hidden_dim_generator,
                            FLAGS.drop_out_prob_generator)
  gen.to(device)
  optimizer = torch.optim.Adam([{
      'params': enc.parameters()
  }, {
      'params': gen.parameters()
  }],
                               lr=FLAGS.learning_rate)
  for i in range(FLAGS.epochs):
    print('-------------\nEpoch {}:\n'.format(i))
    losses = []
    obj_losses = []
    selection_costs = []
    continuity_costs = []
    for _, batch in enumerate(loader):
      x, labels = batch[0].to(device), batch[1].to(device)
      optimizer.zero_grad()
      selection = gen.select(gen(x))
      selection_cost, continuity_cost = gen.loss(selection, x)
      selection_costs.append(selection_cost.tolist())
      continuity_costs.append(continuity_cost.tolist())
      selection = torch.squeeze(selection)
      x = x * selection
      logit = torch.squeeze(enc(x))
      loss = F.mse_loss(logit, labels.float())
      obj_losses.append(loss.item())
      loss += FLAGS.lambda_selection_cost * selection_cost
      loss += FLAGS.lambda_continuity_cost * continuity_cost
      loss.backward()
      losses.append(loss.item())
      optimizer.step()
    print('Loss: ', sum(losses) / len(losses))
    print('Loss for prediction: ', sum(obj_losses) / len(obj_losses))
    print('Selection cost: ', sum(selection_costs)/len(selection_costs))
    print('Continuity cost: ', sum(continuity_costs)/len(continuity_costs))



if __name__ == '__main__':
  app.run(main)
