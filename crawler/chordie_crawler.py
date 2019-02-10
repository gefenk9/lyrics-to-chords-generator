import urllib.request
import os
from html.parser import HTMLParser

class ChordieParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)

    def valid_chords_line(self, line):
        """
        check if the line describes chords
        :param line:
        :return:
        """
        tokens = line.split(' ')
        for chord in tokens:
            if len(chord) > 3:
                return False
        return True

    def chords(self, line):
        chords = []
        if '<u>' in line:
            tokens = line.split('<u>')
            for chord in tokens:
                if '</u>' in chord:
                    chords.append(chord[:chord.index('</u>')])
            return chords
        else:
            return None

    def fix_matching(self, lyrics, chords):
        if lyrics[0].strip() == "":
            lyrics.pop(0)
        if len(lyrics) > len(chords):
            for index, line in enumerate(lyrics):
                if line.strip() == "":
                    chords.insert(index, line)
        return lyrics, chords



    def parse(self, data):
        try:
            chords = []
            lyrics = []
            lines = data.split('\n')
            for line in lines:
                chords_line = self.chords(line)
                if chords_line is not None:
                    st = " ".join(chords_line)
                    chords.append(st)
                else:
                    if len(line.strip()) > 0:
                        if line.strip()[0] is not '<':
                            lyrics.append(line)
                    else:
                        lyrics.append(line)

            lyrics, chords = self.fix_matching(lyrics, chords)

            chords = "\n".join(chords)
            lyrics = "\n".join(lyrics)

            #print("lyrics:\n {} \n chords :\n {}".format(lyrics, chords))
            return lyrics, chords
        except Exception as e:
            print(e.message)

dir = r'C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\pre'
dir_out = r'C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\lyrics_chords'

for root, dirs, files in os.walk(dir):
    for file in files:
        with open(os.path.join(dir, file)) as f:
            data = f.read()

        # Parsing the lyrics and chords from the html tag
        parser = ChordieParser()
        name = file.split('.')[-1]
        lyrics, chords = parser.parse(str(data))

        # Creating the corresponding lyrics and chords files to the song
        with open(os.path.join(dir_out, name+".lyrics"), 'w+') as lyrics_file:
            lyrics_file.write(lyrics)

        with open(os.path.join(dir_out, name+".chords"), 'w+') as chords_file:
            chords_file.write(chords)



