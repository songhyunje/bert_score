### BERT Score for Korean sentence using ETRI BERT (WIP)

PyTorch implementation for BERT Score

- Original code can be found at [bert_score](https://github.com/Tiiiger/bert_score)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)

#### 준비사항 

- PyTorch >= 1.4
- Transformer >= 2.8

```
pip install -r requirements.txt
```

- ETRI BERT cased model ([Korean_BERT_WordPiece](http://aiopen.etri.re.kr/service_dataset.php))

#### 주의사항 
vocab.txt의 전체 개수가 큰 의미가 없어 제거하였음. 따라서, 아래와 같이 vocab.txt이 시작하도록 수정해야 함.
``` 
[PAD] 19863972
[UNK] 19863972
...
```

#### 사용법
```python
from bert_scorer import BertScorer
from utils import get_idf_dict

path = 'path'  # ETRI BERT resource path
scorer = BertScorer(path, batch_size=16)

hyps = ["'렘데시비르' 국내 들여온다...\"중증 환자에 투여\"", "\"국민 프로듀싱을 모의로\"...'프로듀스101' 투표 조작 피디 실형"]
refs = ["'프듀 투표조작' PD 두명에 실형 선고", "'프로듀스 투표조작' 안준영 PD 1심 징역 2년...\"시청자 믿음 저버렸다\""]
scores = scorer.score(hyps, refs)
print(scores)

# add idf scores
idf_dict_ref = get_idf_dict(refs, scorer.tokenizer)
idf_dict_hyp = get_idf_dict(hyps, scorer.tokenizer)
scores = scorer.score_batch(refs, hyps, idf_weight=True, idf_dict_ref=idf_dict_ref, idf_dict_hyp=idf_dict_hyp)
print(scores)
```

#### 수정할 사항
- Rscaling 추가
- Moverscore를 추가
