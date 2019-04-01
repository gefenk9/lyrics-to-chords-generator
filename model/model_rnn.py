import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from model.create_training_data import prep_data
torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def vectors_to_tags(out_vectors, tags_to_ix):
    results = []
    for vec in out_vectors:
        out_ix = (vec==max(vec)).nonzero()[0]
        for tag in tags_to_ix:
            if tags_to_ix[tag] == out_ix:
                results.append(tag)
                break
    return results



class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


word_to_ix, tag_to_ix, training_data = prep_data()
random.shuffle(training_data)
test = training_data[int(0.85*(len(training_data))):]
training_data = training_data[:int(0.85*(len(training_data)))]
EMBEDDING_DIM=6
HIDDEN_DIM=6
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    print (test[1][0])
    inputs = prepare_sequence(test[1][0], word_to_ix)
    tag_scores = model(inputs)
    results = vectors_to_tags(tag_scores, tag_to_ix)
    print(results)

for epoch in range(300):# again, normally you would NOT do 300 epochs, it is toy data
    print("{} %".format(epoch*100.0/300.0))
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    #example = "I need you baby one more time".split(" ")
    inputs = prepare_sequence(test[1][0], word_to_ix)
    tag_scores = model(inputs)
    results = vectors_to_tags(tag_scores, tag_to_ix)

    print(results)
    print(test[1][1])