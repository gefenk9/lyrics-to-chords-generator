# Lyrics to Chords Generator

Generate chords to a given lyrics using LSTM

# Usage

## How to train

## How to run the server

On port 8080:

```bash
python3 server.py 8080
```

## How to generate chords from lyrics

```bash
curl http://localhost:8080/to_chords -d '{"lyrics":"banana is good\npapaya is even better"}' -i
```

```
HTTP/1.0 200 OK
Server: BaseHTTP/0.6 Python/3.6.7
Date: Thu, 25 Apr 2019 12:19:32 GMT

{"chords": ["A", "B", "D"]}%
```
