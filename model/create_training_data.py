import os
dir = r"C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\lyrics_chords"
file = r"all-i-want-for-christmas-is-you"
training_data = []

def prep_data():
    with open(os.path.join(dir, file+".lyrics")) as f:
        for index, line in enumerate(f.readlines()):
            new_tup = ["", ""]
            new_tup[0] = line
            training_data.append(new_tup)

    with open(os.path.join(dir, file + ".chords")) as f:
        for index, line in enumerate(f.readlines()):
            training_data[index][1] = line

    word_to_ix = {}
    tag_to_ix = {}

    for sent, tags in training_data:
        for word in sent.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags.split():
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix