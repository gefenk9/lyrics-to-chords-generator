import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from model.create_training_data import prep_data
import matplotlib.pyplot as plt
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

def check_error(data_set , model, word_to_ix, tag_to_ix, loss_function):
    sum_loss = 0
    for sentence, tags in data_set:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        sum_loss += loss
    return (float(sum_loss) / float(len(data_set)))

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
EMBEDDING_DIM=16
HIDDEN_DIM=32
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


num_of_epochs = 1500
train_errors_per_epoch= []
test_errors_per_epoch= []
for epoch in range(num_of_epochs):# again, normally you would NOT do 300 epochs, it is toy data
    gen_loss = 0
    for sentence, tags in training_data:
        try:
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

            gen_loss += loss.data
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(e)

    trainging_err = check_error(training_data, model, word_to_ix, tag_to_ix, loss_function)
    test_err = check_error(test, model, word_to_ix, tag_to_ix, loss_function)
    train_errors_per_epoch.append(trainging_err)
    test_errors_per_epoch.append(test_err)
    print("finished {} % loss is  {} train err {} test err {}".format(epoch * 100.0 / num_of_epochs, gen_loss / float(len(training_data)),trainging_err ,test_err ))

print ("plotting")
xs = range(len(train_errors_per_epoch))
ys = train_errors_per_epoch
plt.plot(xs, ys, color="blue", label="train loss")
xs = range(len(test_errors_per_epoch))
ys = test_errors_per_epoch
plt.plot(xs, ys, color="red", label="test loss")
plt.savefig('loss_curve')

# See what the scores are after training
with torch.no_grad():
    suc = 0
    for test_sent in test:
        inputs = prepare_sequence(test_sent[0], word_to_ix)
        tag_scores = model(inputs)
        true_result = test_sent[1]
        predicted_results = vectors_to_tags(tag_scores, tag_to_ix)
        if true_result == predicted_results:
            suc += 1

        print("The sent: {}".format(test_sent[0]))
        print("What the model predict: {}".format(predicted_results))
        print("The true result: {}".format(true_result))
    print (suc,float(len(test)))