import os
import random
dir = r"C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\lyrics_chords"

ready_songs = []

def prep_data():
    word_to_ix = {}
    tag_to_ix = {}
    new_training_data = []
    for file_name in os.listdir(dir):
        file = file_name.split('.')[0]
        training_data = []
        with open(os.path.join(dir, file+".lyrics")) as f:
            for index, line in enumerate(f.readlines()):
                new_tup = ["", ""]
                new_tup[0] = line
                training_data.append(new_tup)

        with open(os.path.join(dir, file + ".chords")) as f:
            for index, line in enumerate(f.readlines()):
                try:
                    if index >= len(training_data):
                        training_data.append(["", line])
                except Exception as e:
                    #TODO
                    pass



        for sent, tags in training_data:
            sent, tags = prep_tags(sent, tags)

            new_training_data.append([sent, tags])
            for w in sent:
                if w not in word_to_ix:
                    word_to_ix[w] = len(word_to_ix)

            for t in tags:
                if t not in tag_to_ix:
                    tag_to_ix[t] = len(tag_to_ix)

    return word_to_ix, tag_to_ix, new_training_data



def prep_tags(sent, chords_tags):
    sent_splitted = sent.strip().split(" ")
    chords_tags_splitted = chords_tags.strip().split(" ")
    copied_tags = chords_tags_splitted.copy()
    while len(sent_splitted) > len(chords_tags_splitted):
        chords_tags_splitted.insert(chords_tags_splitted.index(copied_tags[0]), copied_tags[0])
        random.shuffle(copied_tags)
    dense_tags = chords_tags_splitted.copy()
    i = 0
    while len(sent_splitted) < len(dense_tags):
        new_chord = ""
        new_chord = " ".join(dense_tags[i:i+2])
        dense_tags = dense_tags[i+2:]
        dense_tags.insert(0, new_chord)



    return sent_splitted, dense_tags
