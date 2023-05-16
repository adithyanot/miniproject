import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Define the Encoder class
import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.init_hidden(batch_size)
        outputs, hidden = self.lstm(inputs, hidden)
        outputs = outputs.view(batch_size, -1, 2, self.hidden_size)  # outputs shape: [batch_size, sequence_length, 2, hidden_size]
        forward_hidden = outputs[:, -1, 0, :].unsqueeze(0)  # forward hidden state of last time step, shape: [1, batch_size, hidden_size]
        backward_hidden = outputs[:, 0, 1, :].unsqueeze(0)  # backward hidden state of first time step, shape: [1, batch_size, hidden_size]
        hidden_state = torch.cat((forward_hidden, backward_hidden), dim=2)  # concatenate the two hidden states, shape: [1, batch_size, 2*hidden_size]
        return outputs,hidden_state

    def init_hidden(self, batch_size):
        # initialize the hidden states with zeros
        return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size),
                torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))

    
    

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, file_path: str, bpe_path: str, emb_path: str):
        self.data = []
        self.bpe_dict = {}
        with open(bpe_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    values = line.strip().split(' ')
                    if len(values) == 2:
                        key, value = values
                        self.bpe_dict[key] = value
                        
            
                    else:
                        print(f"Skipping line: {line}")
                else:
                    print("Skipping empty line.")
                
              
                

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split(' ')
                tokens = []
                for word in words:
                    if word in ['<s>', '</s>']:
                        tokens.append(word)
                    else:
                        bpe_word = ''
                        for char in word:
                            bpe_word += ' ' + self.bpe_dict.get(char, char)
                        tokens += bpe_word.strip().split(' ')
                self.data.append(tokens)
        self.emb = load_glove_embeddings('glove.6B.300d.txt')

        # np.load(emb_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens = self.data[index]
        emb = np.zeros((len(tokens), 300))
        for i, token in enumerate(tokens):
            emb[i] = self.emb.get(token, np.zeros(300))
        return emb.astype(np.float32)

# Define the main function
def main():
    file_path = 'hi/cm/train.src'
    bpe_path = 'hi/cm/bpe_vocab.txt'
    emb_path = 'glove.6B.300d.txt'
    batch_size = 1
    input_size = 300
    hidden_size = 512
    num_layers = 2

    # Instantiate the encoder
    encoder = BiLSTMEncoder(input_size, hidden_size, num_layers)

    # Load the dataset
    dataset = TextDataset(file_path, bpe_path, emb_path)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Encode the input and store the hidden state representation
    # for batch in dataloader:
    #     batch = batch.to(torch.device('cuda:0'))
    #     hidden_state = encoder(batch)
    #     print(hidden_state)
    
    # for batch in dataloader:
    #     hidden_state = encoder(batch)
    # print(hidden_state)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden = encoder.init_hidden(batch_size)  # Initialize the hidden state
    for batch in dataloader:
        batch_emb = batch.clone().detach().requires_grad_(True)

        batch_emb = batch_emb.to(device)
        outputs, hidden_state = encoder(batch_emb)
        # hidden_state = hidden_state.permute(1, 0, 2) # transpose to expected shape

        # hidden_state = hidden_state.view(encoder.num_layers * 2, batch_size, encoder.hidden_size)#reshape
        hidden_state = hidden_state[-1]  # Use only the final layer's hidden state
    print(hidden_state)

if __name__ == '__main__':
    main()
