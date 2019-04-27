import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from model.create_training_data import prep_data
import matplotlib.pyplot as plt
import glob
import os
import pickle
torch.manual_seed(1)

#Hyper Parameters

EMBEDDING_DIM = 7#4th root of
HIDDEN_DIM = 32
HIDDEN_LAYERS = 3#TRYING
LEARNING_RATE = 1e-3

def prepare_sequence(seq, to_ix):
    for index, w in enumerate(seq):
        if w not in to_ix:
            seq[index] = ''
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

def check_error(data_set , model, word_to_ix, tag_to_ix, to_print=False):
    errors = []
    for sent_chords in data_set:
        inputs = prepare_sequence(sent_chords[0], word_to_ix)
        tag_scores = model(inputs)
        true_result = sent_chords[1]
        predicted_results = vectors_to_tags(tag_scores, tag_to_ix)
        suc = 0
        for index, chord in enumerate(true_result):
            if chord == predicted_results[index]:
                suc += 1
        error_prec = float(suc) / float(len(true_result))
        errors.append(error_prec)
        if to_print:
            print("The sent: {}".format(sent_chords[0]))
            print("{} model predict: {} true result: {}".format(error_prec, predicted_results, true_result))
    return 1 - float(sum(errors)) / float(len(errors))

def check_loss(data_set, model, word_to_ix, tag_to_ix, loss_function):
    sum_loss = 0
    for sentence, tags in data_set:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        sum_loss += loss
    return (float(sum_loss) / float(len(data_set)))

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, hidden_layers):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.hidden_layers, dropout=0)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.autograd.Variable(torch.zeros(self.hidden_layers, 1, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(self.hidden_layers, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def main():
    word_to_ix, tag_to_ix, data = prep_data()
    random.shuffle(data)
    test = data[:int(0.1*(len(data)))]
    training_data = data[int(0.1*(len(data))):]
    validation = training_data[:int(0.2*(len(training_data)))]
    training_data = training_data[int(0.2*(len(training_data))):]

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), HIDDEN_LAYERS)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


    num_of_epochs = 300
    train_loss_per_epoch= []
    validation_loss_per_epoch= []
    test_err_per_epoch = []
    train_err_per_epoch = []
    for epoch in range(num_of_epochs):
        gen_loss = 0
        for sentence, tags in training_data:
            try:
                # Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                model.hidden = model.init_hidden()

                # Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = prepare_sequence(tags, tag_to_ix)

                # Run our forward pass.
                tag_scores = model(sentence_in)

                # Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)

                gen_loss += loss.data
                loss.backward(retain_graph=True)
                optimizer.step()
            except Exception as e:
                print(e)

        training_loss = check_loss(training_data, model, word_to_ix, tag_to_ix, loss_function)
        validation_loss = check_loss(validation, model, word_to_ix, tag_to_ix, loss_function)
        test_err = check_error(test, model, word_to_ix, tag_to_ix)
        train_err = check_error(training_data, model, word_to_ix, tag_to_ix)
        train_loss_per_epoch.append(training_loss)
        validation_loss_per_epoch.append(validation_loss)
        train_err_per_epoch.append(train_err)
        test_err_per_epoch.append(test_err)
        print("finished {} % training loss is  {} training loss {} "
              "validation loss {} test err {}".format(epoch * 100.0 / num_of_epochs,
                                                      float(gen_loss) / float(len(training_data)),
                                                      training_loss, validation_loss, test_err))
    check_error(test, model, word_to_ix, tag_to_ix, True)

    print("plotting loss curve")

    xs = range(len(train_loss_per_epoch))
    ys = train_loss_per_epoch
    plt.plot(xs, ys, color="blue", label="train loss")

    xs = range(len(validation_loss_per_epoch))
    ys = validation_loss_per_epoch
    plt.plot(xs, ys, color="red", label="validation loss")

    plt.legend(loc='upper right', shadow=True)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss Curve')

    plt.savefig('loss_curve_'+time.strftime("%Y%m%d_%H%M%S")+"_hid_layers"+str(HIDDEN_LAYERS)+"_EMBEDDIM"+
                str(EMBEDDING_DIM)+"_HIDDIM"+str(HIDDEN_DIM))

    plt.close()
    xs = range(len(train_err_per_epoch))
    ys = train_err_per_epoch
    plt.plot(xs, ys, color="blue", label="train error")

    xs = range(len(test_err_per_epoch))
    ys = test_err_per_epoch
    plt.plot(xs, ys, color="red", label="test error")

    plt.legend(loc='upper right', shadow=True)
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.title('error Curve')

    plt.savefig('error_curve_'+time.strftime("%Y%m%d_%H%M%S")+"_hid_layers"+str(HIDDEN_LAYERS)+"_EMBEDDIM"+
                str(EMBEDDING_DIM)+"_HIDDIM"+str(HIDDEN_DIM))


    torch.save(model, './../model_versions/model_details_'+time.strftime("%Y%m%d_%H%M%S"))
    with open('word_to_ix', 'wb') as pickle_file:
        pickle.dump(word_to_ix, pickle_file)
    with open('tag_to_ix', 'wb') as pickle_file:
        pickle.dump(tag_to_ix, pickle_file)

def load_existing_model():
    list_of_files = glob.glob('./../model_versions/model_details_*')
    file = max(list_of_files , key = os.path.getctime)
    model = torch.load(file)
    return model

def get_chords(lyrics):
    model = load_existing_model()
    with open('word_to_ix', 'rb') as pickle_file:
        word_to_ix = pickle.load(pickle_file)

    with open('tag_to_ix', 'rb') as pickle_file:
        tag_to_ix = pickle.load(pickle_file)

    chords = []
    for sent in lyrics:
        inputs = prepare_sequence(sent, word_to_ix)
        tag_scores = model(inputs)
        predicted_results = vectors_to_tags(tag_scores, tag_to_ix)
        chords.append(predicted_results)
    return chords
#load_existing_model()

if __name__ == '__main__':
    main()
    #get_chords([['hey','there', 'beautiful', 'girl'],['youre', 'so' ,'beautiful']])

