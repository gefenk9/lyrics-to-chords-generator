# Lyrics to Chords Generator

Generate chords to a given lyrics using LSTM

# Usage

You can access our official user friendly website
https://gefenk9.github.io/lyrics-to-chords-generator/

You should allow mix content via your browser, because the website is served with HTTPS and the server answers with HTTP.

[(Click here for help)](https://pearsonnacommunity.force.com/support/s/article/How-to-display-mixed-content-with-Google-Chrome-Internet-Explorer-or-Firefox-1408394589290)

## How to train

Run model_LSTM to train the model with the current hyperparamters.  
If you wish to edit the hyperparameters you can do it by editing the variables in the code under the comment `#Hyperparameters`.

## How to save & load the trained model

There are a lot of different versions of the model under the `model_versions` directory.

```python
# For loading  existing model
 model = torch.load(file)

 # For saving your own type of model you shoould run :
 torch.save(model, './../model_versions/model_details_'+time.strftime("%Y%m%d_%H%M%S"))
```

## How to run the server

Dependencies:

```
pip3 install torch torchvision matplotlib
```

The to bind tn port 8080:

```bash
python3 server.py 8080
```

## How to generate chords from lyrics

```bash
curl http://localhost:8080/to_chords -d '{"lyrics":"i love you\nlife is even better with you"}' -i
```

```
HTTP/1.0 200 OK
Server: BaseHTTP/0.6 Python/3.6.7
Date: Thu, 25 Apr 2019 12:19:32 GMT

{"chords": [["C","C","G"],["C","C","G","G","G","G"]]}
```

Each inner array coresponds to a lyrics line.
Each chord in an inner array coresponds to a word in the line.
