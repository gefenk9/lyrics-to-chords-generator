# Lyrics to Chords Generator

Generate chords to a given lyrics using LSTM

# Usage

## How to train

run model_LSTM for train the model with the current hyperparamters
if you wish to edit the hyperparameters you can do it by edit the variable sin the code under the comment #Hyperparameters

## How to save & load the trained model

there is a lot of different versions of the model under model_versions file


```python
# For loading  existing model
 model = torch.load(file)
 
 # For saving your own type of model you shoould run : 
 torch.save(model, './../model_versions/model_details_'+time.strftime("%Y%m%d_%H%M%S"))
```

## How to run the server

On port 8080:

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

{"chords": [["C", "C","G"],["C", "C","G","G","G","G"]]}
```
