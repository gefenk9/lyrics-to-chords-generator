import urllib.request
from html.parser import HTMLParser

class ChordieParser(HTMLParser):
    def __init__(self, song_name, out_folder):
        HTMLParser.__init__(self)
        self.song_name = song_name

    def handle_data(self, data):
        if "t:{}".format(self.song_name) in data.lower():
            return (">>> ", data)


opener = urllib.request.FancyURLopener({})
url = "https://www.chordie.com/chord.pere/www.guitaretab.com/b/boyce-avenue/268379.html"
f = opener.open(url)
content = f.read()

parser = ChordieParser('perfect')
l = parser.feed(str(content))
print (l)

