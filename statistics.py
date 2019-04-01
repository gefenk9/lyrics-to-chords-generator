import os

dir = r"C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\lyrics_chords"

all = 0
wrong = 0
for file_name in os.listdir(dir):
    file = file_name.split('.')[0]
    try:
        with open(os.path.join(dir, file + ".lyrics")) as f:
            lyrics_lines = len(f.readlines())
        with open(os.path.join(dir, file + ".chords")) as f:
            chords_lines = len(f.readlines())
    except :
        print("error")
    all += 1
    if chords_lines < 0.8*(lyrics_lines):
        wrong+=1
        os.remove(os.path.join(dir, file + ".lyrics"))
        os.remove(os.path.join(dir, file + ".chords"))

print (wrong, all)
