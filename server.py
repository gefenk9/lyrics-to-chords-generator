#!/usr/bin/env python
 
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import json

class LyricsChordsServer(BaseHTTPRequestHandler):
    def getLyrics(self):
        contentLength = int(self.headers["Content-Length"])
        postData = self.rfile.read(contentLength)
        return json.loads(postData)["lyrics"]

    def do_POST(self):
        if self.path == "/to_chords":
            try:
                lyrics = self.getLyrics()

                # TODO: get chords from model
                mock_answer = {"chords":["A","B","D"]}

                self.send_response(200)
                self.end_headers()

                # TODO: return chords as json 
                self.wfile.write(bytes(json.dumps(mock_answer), "utf8")) 
            except:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(bytes('Input data must be of type JSON: {"lyrics":"bla bla bla"}',"utf8"))

            return
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes("Only HTTP POST to path /to_chords is allowd.","utf8"))
 
print("starting server...")
httpd = HTTPServer(("0.0.0.0", int(sys.argv[1])), LyricsChordsServer)
httpd.serve_forever()
 
 
