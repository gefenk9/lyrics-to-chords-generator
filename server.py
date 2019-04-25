#!/usr/bin/env python
 
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import os
import json

website_dir = os.path.join(os.path.dirname(__file__), 'website','build')

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
                answer = {"chords":["A","B","D"]}

                self.send_response(200)
                self.end_headers()

                self.wfile.write(bytes(json.dumps(answer), "utf8")) 
            except:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(bytes('Input data must be of type JSON: {"lyrics":"bla bla bla"}',"utf8"))

            return
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes("Only HTTP POST to path /to_chords is allowd.","utf8"))

        #Handler for the GET requests
    def do_GET(self):
        if self.path=="/":
            self.path= "index.html"
        if self.path.startswith("/lyrics-to-chords-generator"):
            self.path=self.path.replace("/lyrics-to-chords-generator","",1)

        final_path = website_dir + "/" + self.path

        try:
            #Check the file extension required and
            #set the right mime type

            sendReply = False
            if final_path.endswith(".html"):
                mimetype='text/html'
                sendReply = True
            elif final_path.endswith(".png"):
                mimetype='image/png'
                sendReply = True
            elif final_path.endswith(".js"):
                mimetype='application/javascript'
                sendReply = True
            elif final_path.endswith(".css"):
                mimetype='text/css'
                sendReply = True
            elif final_path.endswith(".woff2"):
                mimetype='font/woff2'
                sendReply = True
            elif final_path.endswith(".json"):
                mimetype='appllication/json'
                sendReply = True

            if sendReply == True:
                with  open(final_path,"rb") as f:
                    self.send_response(200)
                    self.send_header('Content-type',mimetype)
                    self.end_headers()
                    self.wfile.write(f.read())
            return

        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)
 

port = int(sys.argv[1])
print("Starting server on http://localhost:%d/" % port)
httpd = HTTPServer(("0.0.0.0", port), LyricsChordsServer)
httpd.serve_forever()
 
 
