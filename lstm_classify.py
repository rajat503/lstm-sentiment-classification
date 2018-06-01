import os
import numpy as np
import random
import pickle

random.seed(9001)
IMDB_VOCAB_PATH = "./aclImdb/imdb.vocab"
GLOVE_PATH = "./glove.840B.300d.txt"
GLOVE_DIM = 300
TRAIN_DATASET_PATH = "./aclImdb/train/"

hidden_dim = 100

imdb_vocab = set()
with open(IMDB_VOCAB_PATH) as f:
    for line in f:
        imdb_vocab.add(line.strip().lower())

glove_embeddings = {}
with open(GLOVE_PATH) as f:
    for line in f:
        entry = line.split(" ")
        word = entry[0].lower()
        if word in imdb_vocab:
            glove_embeddings[word] = list(map(float, entry[1:]))

def structure_dataset(dir_path):
    dataset = []
    for entry in os.listdir(dir_path+"/pos"):
        content = ""
        with open(dir_path+"/pos/"+entry) as f:
            for line in f:
                content += line
        dataset.append((line, 1))
    for entry in os.listdir(dir_path+"/neg"):
        content = ""
        with open(dir_path+"/neg/"+entry) as f:
            for line in f:
                content += line
        dataset.append((line, 0))
    return dataset

def get_sentence_tensor(sentence):
    tensor = []
    sentence = sentence.lower().split(" ")
    for word in sentence:
        token = word.lower()
        if token in glove_embeddings.keys():
            tensor.append([[glove_embeddings[token]]])
        else:
            tensor.append([[[0.0] * GLOVE_DIM]])
    return np.array(tensor)    

data = structure_dataset(TRAIN_DATASET_PATH)
random.shuffle(data)
train_data = data[:int(0.8*len(data))]
validation_data = data[int(0.8*len(data)):]

import torch.nn as nn
import torch
cuda = torch.device('cuda')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        probs = self.softmax(self.fc(output[0]))
        return probs, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

lstm = LSTM(GLOVE_DIM, hidden_dim, 2)
optimizer = torch.optim.Adam(lstm.parameters())
criterion = nn.NLLLoss()


def train(sentence_tensor, label_tensor):
    hidden = lstm.initHidden()
    optimizer.zero_grad()
    for i in range(sentence_tensor.size()[0]):
        output, hidden = lstm.forward(sentence_tensor[i], hidden)
    loss = criterion(output, label_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()
    
for i in range(len(train_data)):
    print(train(torch.Tensor(get_sentence_tensor(train_data[i][0])), torch.Tensor([train_data[i][1]]).long()))

def evaluate(sentence_tensor):
    hidden = lstm.initHidden()
    optimizer.zero_grad()
    for i in range(sentence_tensor.size()[0]):
        output, hidden = lstm.forward(sentence_tensor[i], hidden)
    return output

count = 0
for i in range(len(validation_data)):
    outp = evaluate(torch.Tensor(get_sentence_tensor(validation_data[i][0]))).data.cpu().numpy()
    if np.argmax(np.exp(outp)) == validation_data[i][1]:
        count += 1
        print(i)
print(count/float(len(validation_data)))
