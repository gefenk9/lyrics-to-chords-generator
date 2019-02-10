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

            chords = "\n".join(chords)
            lyrics = "\n".join(lyrics)

            print("lyrics:\n {} \n chords :\n {}".format(lyrics, chords))
            return lyrics, chords
        except Exception as e:
            print(e.message)



name = r"www.e-chords.com.chords.-luke-phua-kay-kiat-.my-saviour-my-lord"
dir = r'C:\Users\gefenk\Desktop'
with open(os.path.join(dir, name)) as f:
    data = f.read()

parser = ChordieParser()
parser.parse(str(data))




"""
opener = urllib.request.FancyURLopener({})
url = "https://www.chordie.com/chord.pere/www.azchords.com/t/taylorswift-tabs-34955/shakeitoff2-tabs-877676.html"
f = opener.open(url)
content = f.read()

parser = ChordieParser()
parser.feed(str(content))

"""

