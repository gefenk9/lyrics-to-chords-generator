import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from model.create_training_data import prep_data
import matplotlib.pyplot as plt
torch.manual_seed(1)

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs

class Chords_Tagger():
    def __init__(self):
        #prepare the data
        word_to_ix, tag_to_ix, training_data = prep_data()

        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        random.shuffle(training_data)
        self.test_data = training_data[int(0.85 * (len(training_data))):]
        self.training_data = training_data[:int(0.85 * (len(training_data)))]

        self.EMBEDDING_DIM = 24
        self.HIDDEN_DIM = 32
        self.HIDDEN_LAYERS = 3

        self.model = LSTMClassifier(self.EMBEDDING_DIM, self.HIDDEN_DIM, len(word_to_ix), 1, self.HIDDEN_LAYERS)
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

        self.num_of_epochs = 10

    def train_model(self):
        train_errors_per_epoch = []
        test_errors_per_epoch = []
        for epoch in range(self.num_of_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
            gen_loss = 0
            for sent, chords in self.training_data:
                try:
                    self.model.zero_grad()

                    self.model.hidden = self.model.init_hidden()
                    sentence_in = self._prepare_sequence(sent, self.word_to_ix)
                    chords_in = self._prepare_sequence(chords, self.tag_to_ix)

                    input = torch.stack([sentence_in, chords_in])

                    # Step 3. Run our forward pass.
                    tag_scores = self.model(input)

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    loss = self.loss_function(tag_scores, torch.tensor(1))

                    gen_loss += loss.data
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                except Exception as e:
                    print(e)

            trainging_err = self.check_model(self.training_data)
            test_err = self.check_model(self.test)
            train_errors_per_epoch.append(trainging_err)
            test_errors_per_epoch.append(test_err)
            print("finished {} % loss is  {} train err {} test err {}".format((epoch+1) * 100.0 / self.num_of_epochs,
                                                                              gen_loss / float(len(self.training_data)),
                                                                              trainging_err, test_err))
        self.plot_result(train_errors_per_epoch, test_errors_per_epoch)

    def check_model(self, data_set):
        sum_loss = 0
        for sentence, tags in data_set:
            sentence_in = self._prepare_sequence(sentence, self.word_to_ix)
            targets = self._prepare_sequence(tags, self.tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = self.model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = self.loss_function(tag_scores, targets)
            sum_loss += loss
        return (float(sum_loss) / float(len(data_set)))

    def plot_result(self, train_errors_per_epoch, test_errors_per_epoch):
        print("plotting")
        xs = range(len(train_errors_per_epoch))
        ys = train_errors_per_epoch
        plt.plot(xs, ys, color="blue", label="train loss")
        xs = range(len(test_errors_per_epoch))
        ys = test_errors_per_epoch
        plt.plot(xs, ys, color="red", label="test loss")
        plt.savefig('loss_curve')

        with torch.no_grad():
            suc = 0
            for test_sent in self.test_data:
                inputs = self._prepare_sequence(test_sent[0], self.word_to_ix)
                tag_scores = self.model(inputs)
                true_result = test_sent[1]
                predicted_results = self._vectors_to_tags(tag_scores, self.tag_to_ix)
                if true_result == predicted_results:
                    suc += 1

                print("The sent: {}".format(test_sent[0]))
                print("What the model predict: {}".format(predicted_results))
                print("The true result: {}".format(true_result))
            print(suc, float(len(self.test_data)))

        torch.save(self.model, './model_versions/model_details')

    def _prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def _vectors_to_tags(self, out_vectors, tags_to_ix):
        results = []
        for vec in out_vectors:
            out_ix = (vec==max(vec)).nonzero()[0]
            for tag in tags_to_ix:
                if tags_to_ix[tag] == out_ix:
                    results.append(tag)
                    break
        return results


x = Chords_Tagger()
x.train_model()
