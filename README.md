# PromptBERT: Prompt makes BERT Better at Sentence Embeddings

## Setup
Install Dependencies

``` sh
pip install -r requirements.txt
```

Download Data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd ../../..
cd ./data
bash download_wiki.sh
bash download_nli.sh
cd ..
```

Run prefix tuning:

``` sh
./run.sh prefix
```

Run prefix tuning without training:

``` sh
./eval_only.sh result/prefix
```

New arguments are added in `train.py`, from 332-th line

Search `lahelr` for modifications I have made.
