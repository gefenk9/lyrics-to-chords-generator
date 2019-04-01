import os
import pickle
import random
dir = r"C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\lyrics_chords"

ready_songs = []

def is_ok(sent, tags):
    if len(sent) == 0:
        return True
    if '<' in sent or '|' in sent:
        return False
    has_good_chords = False
    for tag in tags:
        if tag is not '' or tag is not ' ':
            if all(ord(c) < 128 for c in tag):
                has_good_chords = True
            else:
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
                    sent, tags = prep_tags(sent_tag)
                    if not is_ok(sent, tags):
                        continue
                    print (sent, tags)
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


    return word_to_ix, tag_to_ix, new_training_data



def prep_tags(sent_chords):
    if sent_chords[0].strip() == '' and sent_chords[1].strip() == '':
        return [''], ['']
    sent_splitted = sent_chords[0].strip().split(" ")
    chords_tags_splitted = list(sent_chords[1].strip())
    while len(sent_splitted) != len(chords_tags_splitted):
        if len(chords_tags_splitted) > len(sent_splitted):
            if ' ' in chords_tags_splitted:
                chords_tags_splitted.remove(' ')
            else:
                chords_tags_splitted = chords_tags_splitted[:-1]
        else:
            chords_tags_splitted.append(chords_tags_splitted[-1])
    return sent_splitted, chords_tags_splitted
