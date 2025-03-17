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
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'


# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):

  def __init__(self, input_dim, h):  # Add relevant parameters
    super(RNN, self).__init__()
    self.h = h
    self.numOfLayer = 1
    self.rnn = nn.RNN(
      input_dim,
      h,
      self.numOfLayer,
      nonlinearity='tanh',
    )
    self.W = nn.Linear(h, 5)
    self.softmax = nn.LogSoftmax(dim=1)
    self.loss = nn.NLLLoss()

  def compute_Loss(self, predicted_vector, gold_label):
    return self.loss(predicted_vector, gold_label)

  def forward(self, inputs):
    # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
    #rnn_output, _ = self.rnn(inputs)

    # [to fill] obtain output layer representations
    # [to fill] sum over output

    # [to fill] obtain probability dist.

    # option a: resulted about .42 validation accuracy
    # rnn_output, _ = self.rnn(inputs)
    # output = self.W(rnn_output[-1])
    # predicted_vector = self.softmax(output)
    # return predicted_vector

    # option b:
    output, hidden = self.rnn(inputs)  # Step 1: RNN hidden states
    summed_output = output.sum(dim=0)  # Step 3: Sum over output
    logits = self.W(summed_output)  # Step 2: Linear layer
    predicted_vector = torch.nn.functional.softmax(
      logits, dim=-1
    )  # Step 4: Softmax for probabilities
    return predicted_vector

    # option c: tested 0.375, but overfit after 2 epochs
    # # Obtain hidden layer representation
    # _, hidden = self.rnn(inputs)

    # # Obtain output layer representation
    # output = self.W(hidden.squeeze(0))

    # # Compute probability distribution using log softmax
    # predicted_vector = self.softmax(output)

    # return predicted_vector


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
  with open(train_data) as training_f:
    training = json.load(training_f)
  with open(val_data) as valid_f:
    validation = json.load(valid_f)
  if test_data:
    with open(test_data) as test_f:
      test = json.load(test_f)

  tra = []
  val = []
  tst = []
  for elt in training:
    tra.append((elt["text"].split(), int(elt["stars"] - 1)))
  for elt in validation:
    val.append((elt["text"].split(), int(elt["stars"] - 1)))
  if test_data:
    for elt in test:
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

  print("========== Loading data ==========")
  train_data, valid_data, test_data = load_data(
    args.train_data, args.val_data, args.test_data
  )  # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

  # Think about the type of function that an RNN describes. To apply it,
  #   you will need to convert the text data into vector representations.
  # Further, think about where the vectors will come from. There are 3 reasonable choices:
  # 1) Randomly assign the input to vectors and learn better embeddings during training;
  #    see the PyTorch documentation for guidance
  # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of
  #    {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
  # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
  # Option 3 will be the most time consuming, so we do not recommend starting with this

  print("========== Vectorizing data ==========")
  model = RNN(50, args.hidden_dim)  # Fill in parameters
  #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

  stopping_condition = False
  epoch = 0

  last_train_accuracy = 0
  last_validation_accuracy = 0

  epoch_results_strings = []
  epoch_training_losses = []

  while not stopping_condition:
    epoch_total_loss = 0
    random.shuffle(train_data)
    model.train()
    # You will need further code to operationalize training, ffnn.py may be helpful
    print("Training started for epoch {}".format(epoch + 1))

    train_data = train_data
    correct = 0
    total = 0
    minibatch_size = 16
    N = len(train_data)

    loss_total = 0
    loss_count = 0
    for minibatch_index in tqdm(range(N // minibatch_size)):
      optimizer.zero_grad()
      loss = None
      for example_index in range(minibatch_size):
        input_words, gold_label = train_data[minibatch_index * minibatch_size +
                                             example_index]
        input_words = " ".join(input_words)

        # Remove punctuation
        input_words = input_words.translate(
          input_words.maketrans("", "", string.punctuation)
        ).split()

        # Look up word embedding dictionary
        vectors = [
          word_embedding[i.lower()]
          if i.lower() in word_embedding.keys() else word_embedding['unk']
          for i in input_words
        ]
        vectors = np.array(vectors)
        # Transform the input into required shape
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)

        # Get loss
        example_loss = model.compute_Loss(
          output.view(1, -1), torch.tensor([gold_label])
        )

        # Get predicted label
        predicted_label = torch.argmax(output)

        correct += int(predicted_label == gold_label)
        #print(predicted_label, gold_label)
        total += 1
        if loss is None:
          loss = example_loss
        else:
          loss += example_loss

      loss = loss / minibatch_size

      epoch_total_loss += loss.data

      loss_total += loss.data
      loss_count += 1

      loss.backward()
      optimizer.step()

    print(loss_total / loss_count)
    training_loss = epoch_total_loss / (N // minibatch_size)
    epoch_training_losses.append(training_loss)
    print("Training completed for epoch {}".format(epoch + 1))
    print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
    trainning_accuracy = correct / total

    model.eval()
    correct = 0
    total = 0
    random.shuffle(valid_data)
    print("Validation started for epoch {}".format(epoch + 1))
    valid_data = valid_data

    for input_words, gold_label in tqdm(valid_data):
      input_words = " ".join(input_words)
      input_words = input_words.translate(
        input_words.maketrans("", "", string.punctuation)
      ).split()
      vectors = [
        word_embedding[i.lower()]
        if i.lower() in word_embedding.keys() else word_embedding['unk']
        for i in input_words
      ]
      vectors = np.array(vectors)
      vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
      output = model(vectors)
      predicted_label = torch.argmax(output)
      correct += int(predicted_label == gold_label)
      total += 1
      # print(predicted_label, gold_label)
    print("Validation completed for epoch {}".format(epoch + 1))
    print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
    validation_accuracy = correct / total
    epoch_results_strings.append(
      f"Epoch {epoch + 1} Training Loss: {training_loss} Training accuracy: {correct / total}, Validation accuracy: {correct / total}\n"
    )

    if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
      stopping_condition = True
      print("Training done to avoid overfitting!")
      print("Best validation accuracy is:", last_validation_accuracy)
    else:
      last_validation_accuracy = validation_accuracy
      last_train_accuracy = trainning_accuracy

    epoch += 1
    if epoch >= args.epochs:
      break

  if args.do_train:
    print("========== Testing ==========")
    # fill in testing function
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    minibatch_size = 16
    N = len(test_data)
    for minibatch_index in tqdm(range(N // minibatch_size)):
      loss = None
      for example_index in range(minibatch_size):
        input_words, gold_label = test_data[minibatch_index * minibatch_size +
                                            example_index]
        input_words = " ".join(input_words)
        input_words = input_words.translate(
          input_words.maketrans("", "", string.punctuation)
        ).split()
        vectors = [
          word_embedding[i.lower()]
          if i.lower() in word_embedding.keys() else word_embedding['unk']
          for i in input_words
        ]
        vectors = np.array(vectors)
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)

        # get loss
        example_loss = model.compute_Loss(
          output.view(1, -1), torch.tensor([gold_label])
        )
        predicted_label = torch.argmax(output)
        correct += int(predicted_label == gold_label)
        total += 1
        if loss is None:
          loss = example_loss
        else:
          loss += example_loss
      loss = loss / minibatch_size

    print("Testing completed")
    test_accuracy = correct / total
    print("Testing accuracy: {}".format(test_accuracy))
    print("Testing time: {}".format(time.time() - start_time))

  # You may find it beneficial to keep track of training accuracy or training loss;

  # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
  # write out all results to results/test.out

  model_params_string = f"Hidden dimensions: {args.hidden_dim}, Epochs: {args.epochs}\n"
  with open(f"results/test-rnn-{args.hidden_dim}-{args.epochs}.out", "w") as f:
    f.write(model_params_string)
    f.write("========== Epoch results ==========\n")
    f.write(" ".join(epoch_results_strings))
    f.write("\n")
    if args.do_train:

      f.write("========== Testing ==========\n")
      f.write(f"Testing accuracy: {test_accuracy}\n")
    f.write(f"Average loss: {loss}\n")
    #f.write(f"Average error: {avg_error}\n")
    f.write(f"Total time: {(time.time() - program_start_time):.2f}\n")
    f.write(f"========== Model parameters ==========\n")
    #f.write(f"Training accuracy: {correct / total}\n")
    #f.write(f"Validation accuracy: {correct / total}\n")
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
    #f.write(f"Vocab size: {len(vocab)}\n")
    f.write(f"Train data size: {len(train_data)}\n")
    f.write(f"Validation data size: {len(valid_data)}\n")
    #f.write(f"Test data size: {len(test_data)}\n")
    f.write(f"Minibatch size: 16\n")
