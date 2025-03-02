# Mensa meal text embeddings

# Prerequisits
### create venv
```bash
python -m venv venv
source venv/bin/activate
```

### install dependencies
```bash
pip install -r requirements.txt
```

# train finetuned model
```bash
python -m training.main
```

# generate embeddings
```bash
python similarity_embeddings.py
```

# query meal
## query directly
```bash
python read.py "kartoffeleintopf"
```
or, for similarity > x%
```bash
python read.py "kartoffeleintopf" 80
```

## run api
```bash
python api.py
```

## query a meal
```bash
curl -G "http://localhost:5566/similar-threshold" \
  --data-urlencode "query=bohneneintopf brötchen" \
  --data-urlencode "t=70"
```
