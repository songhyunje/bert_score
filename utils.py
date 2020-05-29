from math import log

from collections import defaultdict, Counter


def get_idf_dict(sents, tokenizer):
    idf_count = Counter()
    num_sents = len(sents)
    for sent in sents:
        idf_count.update(set(tokenizer.tokenize(sent)))

    idf_dict = defaultdict(lambda: log((num_sents + 1) / 1))
    idf_dict.update({idx: log((num_sents + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict
