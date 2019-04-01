import urllib.request
import os
import pickle
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
        line = line[line.index('<u>'):]
        while '<u>' in line:
            line = line[:line.index('<u>')] + line[line.index('<u>')+3:]
        while '</u>' in line:
            line = line[:line.index('</u>')] + line[line.index('</u>')+4:]
        return line

    def fix_matching(self, lyrics, chords):
        if lyrics[0].strip() == "":
            lyrics.pop(0)
        if len(lyrics) > len(chords):
            for index, line in enumerate(lyrics):
                if line.strip() == "":
                    chords.insert(index, line)
        return lyrics, chords

    # def parse_line(self, line, next_line):
    #     if '<u>' in line:
    #         # its a chords line
    #         chords_line = self.chords(line)
    #         if chords_line is not '':
    #             return chords_line
    #     else:
    #         # might be a lyrics line
    #         if len(line.strip()) > 0:
    #             if line.strip()[0] is not '<':
    #                 to_add[0] = line
    #         else:
    #             to_add[0] = line
    #


    def parse(self, data):
        try:
            lyrics_chords = []
            lines = data.split('\n')
            last_chords = None
            new_chords = None
            lyrics = None
            for index, line in enumerate(lines):
                if '<u>' in line:
                    # its chords line
                    if new_chords is not None and new_chords is not '':
                        last_chords = new_chords
                    new_chords = self.chords(line)
                    if last_chords is None:
                        last_chords = new_chords

                else:
                    # its lyrics
                    if len(line.strip()) > 0:
                        if '<' in line.strip()[0]:
                            continue
                        else:
                            lyrics = line
                            if new_chords is not '':
                                lyrics_chords.append([lyrics, new_chords])
                            else:
                                lyrics_chords.append([lyrics, last_chords])
                    else:
                        lyrics_chords.append([line, ''])


            return lyrics_chords
            #
            # lyrics, chords = self.fix_matching(lyrics, chords)
            #
            # chords = "\n".join(chords)
            # lyrics = "\n".join(lyrics)

            #print("lyrics:\n {} \n chords :\n {}".format(lyrics, chords))
            #return lyrics, chords
        except Exception as e:
            print(e)

dir = r'C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\pre'
dir_out = r'C:\Users\gefenk\Documents\Deep Learning\chords_to_lyrics_generator\data\lyrics_chords'
nums = 0
for root, dirs, files in os.walk(dir):
    for file in files:
        if nums < 5000:
            try:
                with open(os.path.join(dir, file)) as f:
                    data = f.read()

                # Parsing the lyrics and chords from the html tag
                parser = ChordieParser()
                name = file.split('.')[-1]
                lyrics_chords = parser.parse(str(data))

                # # Creating the corresponding lyrics and chords files to the song
                # with open(os.path.join(dir_out, name+".lyrics"), 'w+') as lyrics_file:
                #     lyrics_file.write(lyrics)
                #
                with open(os.path.join(dir_out, name+".df"), 'wb') as chords_file:
                    pickle.dump(lyrics_chords, chords_file)

                nums+=1
            except Exception as e:
                print (e)



