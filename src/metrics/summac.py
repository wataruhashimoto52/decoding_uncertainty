import numpy as np
# from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from typing import List, Dict
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


class SUMMACMetric(GenerationMetric):
    """
    Calculates SummaC score between model-generated texts and ground truth texts.
    """

    def __init__(self, model_card = 'microsoft/deberta-base-mnli', depend=["greedy_texts"]):
        """
        Parameters:
            rouge_name (str): rouge metric type. Possible values:
                * rouge1
                * rouge2
                * rougeL

            model_card (str): the NLI model used for hallucination evaluation

        """
        super().__init__(depend, "sequence")
        


        # model_card = 'tals/albert-xlarge-vitaminc-mnli'
        self.model_card = model_card
        self.sent_toknizer = AutoTokenizer.from_pretrained(model_card)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(model_card).eval()
        self.sent_model = self.sent_model.cuda().half()

        if model_card == 'microsoft/deberta-base-mnli':
            self.entailment_idx = 2
            self.contradiction_idx = 0
        elif model_card == 'tals/albert-xlarge-vitaminc-mnli':
            self.entailment_idx = 0
            self.contradiction_idx = 1

    def __str__(self):
        return f"summac_cardname_{self.model_card}"

    def _get_nli_socre(self, pd_sent, gt_sent, tokenizer, model):

        # concate_sent_list = [(gt_sent, pd_sent), (gt_sent, gt_sent)]
        concate_sent_list = [(gt_sent, pd_sent)]

        print(f"len(concate_sent_list)={len(concate_sent_list)}")

        batch_tokens = tokenizer.batch_encode_plus(concate_sent_list, padding=True,
                                                        truncation=True, max_length=512,
                                                        return_tensors="pt", truncation_strategy="only_first")
        with torch.no_grad():
            model_outputs = model(**{k: v.cuda() for k, v in batch_tokens.items()})

        batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
        batch_evids = batch_probs[:, self.entailment_idx]
        batch_conts = batch_probs[:, self.contradiction_idx]

        res =  batch_evids - batch_conts # using the direction that is similar to Rouge and BERTScore



        print('RES is: ', res)
        
        final_res = res.cpu().float()[0].item()

        print("summac_res: ", final_res)
        return final_res

    # def _inter_cal_fai_onesample(self, ):

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
        white=None,
    ) -> np.ndarray:
        """
        Calculates SummaC score between model-generated texts and ground truth texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of Rouge Scores for each sample in input.
        """
        if white:
            greedy_text_key = "greedy_texts"
        else:
            greedy_text_key = "blackbox_greedy_texts"
        return np.array(
            [
                
                self._get_nli_socre(hyp, ref, self.sent_toknizer, self.sent_model)
                for hyp, ref in zip(stats[greedy_text_key], target_texts)
            ]
        )