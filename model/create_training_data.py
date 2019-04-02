import os
import pickle
import random
dir = r"C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\lyrics_chords"

ready_songs = []

def is_ok(sent, tags):
    if len(sent) == 0:
        return True
    for word in sent:
        if '<' in word or '|' in word:
            return False
    has_good_chords = False
    for tag in tags:
        if tag is not '' or tag is not ' ':
            if tag.isalpha():
                has_good_chords = True
            else:
                if tag.isnumeric():
                    continue
                return False
    return has_good_chords

def prep_data():
    word_to_ix = {}
    tag_to_ix = {}
    new_training_data = []
    nums = 0
    for file_name in os.listdir(dir):
        try:
            if nums < 100:
                file = file_name.split('.')[0]
                training_data = []
                with open(os.path.join(dir, file+".df"), "rb+") as f:
                    training_data = pickle.load(f)

                for sent_tag in training_data:
                    if sent_tag[0] is None or sent_tag[1] is None:
                        continue
                    if not all(ord(c) < 128 for c in sent_tag[0]):
                        continue

                    sent, tags = prep_tags(sent_tag)
                    if not is_ok(sent, tags):
                        continue
                    #print(sent, tags)
                    new_training_data.append([sent, tags])
                    for w in sent:
                        if w not in word_to_ix:
                            word_to_ix[w] = len(word_to_ix)

                    for t in tags:
                        if t not in tag_to_ix:
                            tag_to_ix[t] = len(tag_to_ix)

                nums+=1
            else:
                break
        except Exception as e:
            print(e)

    print ("The length of training data : {}".format(len(new_training_data)))
    return word_to_ix, tag_to_ix, new_training_data



def prep_tags(sent_chords):
    if sent_chords[0].strip() == '' and sent_chords[1].strip() == '':
        return [''], ['']
    sent_splitted = sent_chords[0].strip().split(" ")
    chords_tags_splitted = list(sent_chords[1].strip().split(" "))
    while len(sent_splitted) != len(chords_tags_splitted):
        if len(chords_tags_splitted) > len(sent_splitted):
            if ' ' in chords_tags_splitted:
                chords_tags_splitted.remove(' ')
            elif '' in chords_tags_splitted:
                chords_tags_splitted.remove('')
            else:
                chords_tags_splitted = chords_tags_splitted[:-1]
        else:
            chords_tags_splitted.append(chords_tags_splitted[-1])
    last_not_empty = chords_tags_splitted[0]
    if last_not_empty == '' or last_not_empty == ' ':
        for tag in chords_tags_splitted:
            if tag is not '' and tag is not ' ':
                last_not_empty = tag
    for index, tag in enumerate(chords_tags_splitted):
        if tag == '' or tag == ' ':
            chords_tags_splitted[index] = last_not_empty
        else:

            last_not_empty = tag

    return sent_splitted, chords_tags_splitted
