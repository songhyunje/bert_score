import logging

import torch

from transformers import BertModel
from tokenization_etri import BertTokenizer

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)


class BertScorer(object):
    def __init__(self, model_name_or_path=None, device='cuda'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.device = device
        self.model.to(self.device)

    def get_bert_embedding(self, sent, max_len=None):
        tokens = self.tokenizer.tokenize(sent)
        logging.debug(tokens)
        encoded = self.tokenizer.encode_plus(tokens,
                                             add_special_tokens=True,
                                             max_length=max_len,
                                             pad_to_max_length=True,
                                             return_tensors="pt")
        for k, v in encoded.items():
            encoded[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=encoded['input_ids'],
                                 attention_mask=encoded['attention_mask'])

        return outputs[0], tokens, encoded['attention_mask']

    def score(self, refs, hyps, idf_dict_ref=None, idf_dict_hyp=None,
              idf_weight=True, max_len=300, rescale_with_baseline=False):
        scores = []
        for ref, hyp in zip(refs, hyps):
            ref_embedding, ref_tokens, ref_masks = self.get_bert_embedding(ref, max_len=max_len)
            hyp_embedding, hyp_tokens, hyp_masks = self.get_bert_embedding(hyp, max_len=max_len)

            # normalized
            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

            sim = torch.bmm(ref_embedding, hyp_embedding.transpose(1, 2))
            masks = torch.bmm(ref_masks.unsqueeze(2).float(), hyp_masks.unsqueeze(1).float()).float()
            sim = sim * masks

            rlen = len(ref_tokens)
            hlen = len(hyp_tokens)
            logger.debug('reference len: %d, hypothesis len: %d', rlen, hlen)
            sim = sim.squeeze(0)[1:rlen + 1, 1:hlen + 1]  # remove special tokens
            precision = sim.max(dim=0)[0]
            recall = sim.max(dim=1)[0]

            if idf_weight and idf_dict_ref and idf_dict_hyp:
                hyp_idf = torch.FloatTensor([idf_dict_hyp[token] for token in hyp_tokens[:max_len - 1]]).to(
                    precision.device)
                hyp_idf.div_(hyp_idf.sum(dim=0))

                ref_idf = torch.FloatTensor([idf_dict_ref[token] for token in ref_tokens[:max_len - 1]]).to(
                    recall.device)
                ref_idf.div_(ref_idf.sum(dim=0))

                P = (precision * hyp_idf).sum(dim=0)
                R = (recall * ref_idf).sum(dim=0)
            elif idf_weight and idf_dict_ref:
                ref_idf = torch.FloatTensor([idf_dict_ref[token] for token in ref_tokens[:max_len - 1]]).to(
                    recall.device)
                ref_idf.div_(ref_idf.sum(dim=0))

                P = precision.sum(dim=0) / hlen
                R = (recall * ref_idf).sum(dim=0)
            else:
                P = precision.sum(dim=0) / hlen
                R = recall.sum(dim=0) / rlen

            logger.debug('Precision: %f, Recall: %f', P, R)
            F = 2 * P * R / (P + R)
            scores.append(F.item())

        return scores


