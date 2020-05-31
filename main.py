from bert_scorer import BertScorer

path = '../resources/bert-etri-cased'
scorer = BertScorer(path, batch_size=16)

hyps = ["'렘데시비르' 국내 들여온다...\"중증 환자에 투여\"", "\"국민 프로듀싱을 모의로\"...'프로듀스101' 투표 조작 피디 실형"]
refs = ["'프듀 투표조작' PD 두명에 실형 선고", "'프로듀스 투표조작' 안준영 PD 1심 징역 2년...\"시청자 믿음 저버렸다\""]
print(scorer.score(hyps, refs))
