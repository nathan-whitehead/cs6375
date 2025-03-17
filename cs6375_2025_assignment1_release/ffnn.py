import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'


# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):

  def __init__(self, input_dim, h):
    super(FFNN, self).__init__()
    self.h = h
    self.W1 = nn.Linear(input_dim, h)
    self.activation = nn.ReLU(
    )  # The rectified linear unit; one valid choice of activation function
    self.output_dim = 5
    self.W2 = nn.Linear(h, self.output_dim)

    self.softmax = nn.LogSoftmax(
    )  # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
    self.loss = nn.NLLLoss(
    )  # The cross-entropy/negative log likelihood loss taught in class

  def compute_Loss(self, predicted_vector, gold_label):
    return self.loss(predicted_vector, gold_label)

  def forward(self, input_vector):
    # [to fill] obtain first hidden layer representation
    hidden = self.activation(self.W1(input_vector))
    # [to fill] obtain output layer representation
    output = self.W2(hidden)
    # [to fill] obtain probability dist.
    predicted_vector = self.softmax(output)
    return predicted_vector


# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
  vocab = set()
  for document, _ in data:
    for word in document:
      vocab.add(word)
  return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
  vocab_list = sorted(vocab)
  vocab_list.append(unk)
  word2index = {}
  index2word = {}
  for index, word in enumerate(vocab_list):
    word2index[word] = index
    index2word[index] = word
  vocab.add(unk)
  return vocab, word2index, index2word


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
  vectorized_data = []
  for document, y in data:
    vector = torch.zeros(len(word2index))
    for word in document:
      index = word2index.get(word, word2index[unk])
      vector[index] += 1
    vectorized_data.append((vector, y))
  return vectorized_data


def load_data(train_data, val_data, test_data=None):
  train_stars = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
  with open(train_data) as training_f:
    training = json.load(training_f)
    for line in training:
      train_stars[str(int(line["stars"]))] += 1
  print(f"Training distribution: {train_stars}")

  val_stars = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
  with open(val_data) as valid_f:
    validation = json.load(valid_f)
    for line in validation:
      val_stars[str(int(line["stars"]))] += 1
  print(f"Validation distribution: {val_stars}")

  test_stars = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
  if test_data:
    with open(test_data) as test_f:
      test = json.load(test_f)
      for line in test:
        test_stars[str(int(line["stars"]))] += 1
    print(f"Test distribution: {test_stars}")

  tra = []
  val = []
  tst = []
  for elt in training:
    tra.append((elt["text"].split(), int(elt["stars"] - 1)))
  for elt in validation:
    val.append((elt["text"].split(), int(elt["stars"] - 1)))
  for elt in test:
    # train/val is 1-3, but test is 3-5... so -3
    tst.append((elt["text"].split(), int(elt["stars"] - 1)))

  return tra, val, tst


if __name__ == "__main__":
  program_start_time = time.time()
  parser = ArgumentParser()
  parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
  parser.add_argument(
    "-e", "--epochs", type=int, required=True, help="num of epochs to train"
  )
  parser.add_argument("--train_data", required=True, help="path to training data")
  parser.add_argument("--val_data", required=True, help="path to validation data")
  parser.add_argument("--test_data", default="to fill", help="path to test data")
  parser.add_argument('--do_train', action='store_true')
  args = parser.parse_args()

  # fix random seeds
  random.seed(42)
  torch.manual_seed(42)

  # load data
  print("========== Loading data ==========")
  train_data, valid_data, test_data = load_data(
    args.train_data, args.val_data, args.test_data
  )  # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
  vocab = make_vocab(train_data)
  vocab, word2index, index2word = make_indices(vocab)

  print("========== Vectorizing data ==========")

  train_data = convert_to_vector_representation(train_data, word2index)
  valid_data = convert_to_vector_representation(valid_data, word2index)
  test_data = convert_to_vector_representation(test_data, word2index)

  model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
  #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  print("========== Training for {} epochs ==========".format(args.epochs))
  epoch_results_strings = []
  epoch_training_losses = []
  for epoch in range(args.epochs):
    epoch_total_loss = 0
    model.train()
    optimizer.zero_grad()
    loss = None
    correct = 0
    total = 0
    start_time = time.time()
    print("Training started for epoch {}".format(epoch + 1))
    random.shuffle(train_data)  # Good practice to shuffle order of training data
    minibatch_size = 16
    N = len(train_data)
    for minibatch_index in tqdm(range(N // minibatch_size)):
      optimizer.zero_grad()
      loss = None
      for example_index in range(minibatch_size):
        input_vector, gold_label = train_data[minibatch_index * minibatch_size +
                                              example_index]
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)
        correct += int(predicted_label == gold_label)
        total += 1
        example_loss = model.compute_Loss(
          predicted_vector.view(1, -1), torch.tensor([gold_label])
        )
        if loss is None:
          loss = example_loss
        else:
          loss += example_loss
      loss = loss / minibatch_size
      epoch_total_loss += loss.item()
      training_loss = loss.item()
      loss.backward()
      optimizer.step()
    training_loss = epoch_total_loss / (N // minibatch_size)
    epoch_training_losses.append(training_loss)

    print("Training completed for epoch {}".format(epoch + 1))
    print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
    print(f"Training time for this epoch: {(time.time() - start_time):.2f}")

    loss = None
    correct = 0
    total = 0
    start_time = time.time()
    print("Validation started for epoch {}".format(epoch + 1))
    minibatch_size = 16
    N = len(valid_data)
    for minibatch_index in tqdm(range(N // minibatch_size)):
      optimizer.zero_grad()
      loss = None
      for example_index in range(minibatch_size):
        input_vector, gold_label = valid_data[minibatch_index * minibatch_size +
                                              example_index]
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)
        correct += int(predicted_label == gold_label)
        total += 1
        example_loss = model.compute_Loss(
          predicted_vector.view(1, -1), torch.tensor([gold_label])
        )
        if loss is None:
          loss = example_loss
        else:
          loss += example_loss
      loss = loss / minibatch_size
    print(f"Validation completed for epoch {epoch + 1} / {args.epochs}")
    print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
    print(f"Validation time for this epoch: {(time.time() - start_time):.2f}")
    epoch_results_strings.append(
      f"Epoch {epoch + 1} Training loss: {training_loss} Training accuracy: {correct / total}, Validation accuracy: {correct / total}\n"
    )

  # now test the model
  if args.do_train:
    print("========== Testing =========")
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    print("Testing started")
    minibatch_size = 16
    N = len(test_data)
    for minibatch_index in tqdm(range(N // minibatch_size)):
      optimizer.zero_grad()
      loss = None
      for minibatch_index in tqdm(range(N // minibatch_size)):
        optimizer.zero_grad()
        loss = None
        for example_index in range(minibatch_size):
          input_vector, gold_label = test_data[minibatch_index * minibatch_size +
                                               example_index]
          predicted_vector = model(input_vector)
          predicted_label = torch.argmax(predicted_vector)
          correct += int(predicted_label == gold_label)
          total += 1
          example_loss = model.compute_Loss(
            predicted_vector.view(1, -1), torch.tensor([gold_label])
          )
          if loss is None:
            loss = example_loss
          else:
            loss += example_loss
        loss = loss / minibatch_size

    print("Testing completed")
    test_accuracy = correct / total if total > 0 else 0
    print("Testing accuracy: {}".format(test_accuracy))
    print(f"Testing time: {(time.time() - start_time):.2f}")

  # ==============================

  # write out all results to results/test.out
  model_params_string = f"Hidden dimensions: {args.hidden_dim}, Epochs: {args.epochs}\n"
  with open(f"results/test-ffnn-{args.hidden_dim}-{args.epochs}.out", "w") as f:
    f.write(model_params_string)
    f.write("========== Epoch results ==========\n")
    f.write(" ".join(epoch_results_strings))
    f.write("\n")
    f.write(f"========== Model parameters ==========\n")
    f.write(f"Hidden dim: {args.hidden_dim}\n")
    f.write(f"Epochs: {args.epochs}\n")
    f.write(f"Train data: {args.train_data}\n")
    f.write(f"Validation data: {args.val_data}\n")
    if args.test_data:
      f.write(f"Test data: {args.test_data}\n")
    f.write(f"Seed: 42\n")
    f.write(f"Optimizer: SGD\n")
    f.write(f"Learning rate: 0.01\n")
    f.write(f"Momentum: 0.9\n")
    f.write(f"Loss: NLLLoss\n")
    f.write(f"Activation: ReLU\n")
    f.write(f"Softmax: LogSoftmax\n")
    f.write(f"Model: FFNN\n")
    f.write(f"Vocab size: {len(vocab)}\n")
    f.write(f"Train data size: {len(train_data)}\n")
    f.write(f"Validation data size: {len(valid_data)}\n")
    if args.test_data:
      f.write(f"Test data size: {len(test_data)}\n")
    f.write(f"========== Test Results ==========\n")
    if args.do_train:
      f.write(f"Testing accuracy: {test_accuracy}\n")
      f.write(f"Average loss: {loss}\n")
