import urllib.request
from html.parser import HTMLParser

class ChordieParser(HTMLParser):
    def handle_data(self, data):
        if "t:" in data.lower() and "st:" in data.lower():
            lyrics, chords = parse_chords(data)
            print(">>> ", chords)

def parse_chords(data):
    lines = data.split('\\n')
    song_artist_name = "{}-{}".format(lines[0], lines[1])
    print (lines)
    chords = lines[2:]
    return chords, chords


opener = urllib.request.FancyURLopener({})
url = "https://www.chordie.com/chord.pere/www.azchords.com/t/taylorswift-tabs-34955/shakeitoff2-tabs-877676.html"
f = opener.open(url)
content = f.read()

parser = ChordieParser()
parser.feed(str(content))

