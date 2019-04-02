#!/bin/bash

# Get list of all artists into artists.txt (urls)
echo {a..z} | 
	tr ' ' '\n' | 
	parallel -P 1 --bar $'curl -L -s --compressed https://www.e-chords.com/browse/{} | grep -Fa \'class="alphabet">\' | grep -Po \'href=".+?"\' | awk -F \'"\' \'{print $2}\' | tail -n +2' > artists.txt

# From artists, get list of all songs into songs.txt (also urls)
cat artists.txt |
	parallel --bar $'curl -L -s --compressed "{}" | grep -Po \'href="http.+?"\' | grep "/chords/{/}/" | sort -u | awk -F \'"\' \'{print $2}\'' > songs.txt

mkdir -p htmls

# From songs.txt, download all the songs pages into the htmls directory
sort -u songs.txt | 
	grep '/chords/' |
	parallel --bar 'curl -L -s --compressed "{}" > htmls/"$(echo "{}" | sed 's/https...//g' | tr / .)"'

mkdir -p pre

# From the htmls files of the songs, extract the lyrics+chords part of the page (it's inside the <pre> html element)
find htmls -type f -print0 | 
	parallel --bar -0 $'awk \'/<pre id="core"/{ok=1} ok{print} ok&&/<\\/pre>/{ok=0}\' < "{}" > "pre/{/}"'
