import numpy as np
# from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from typing import List, Dict
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
# from ctc_score.scorer import Scorer
from .ctc_score.factual_consistency_scorer import FactualConsistencyScorer



class CTCFactConsistency(GenerationMetric):
    """
    Calculates CTC score between model-generated texts and ground truth texts.
    """

    def __init__(self, depend=["greedy_texts"], set_align='D-mix-albert'):
        """
        Parameters:
            rouge_name (str): rouge metric type. Possible values:
                * rouge1
                * rouge2
                * rougeL

            model_card (str): the NLI model used for hallucination evaluation

        """
        super().__init__(depend, "sequence")
        # self.rouge_name = rouge_name
        # self.scorer = rouge_scorer.RougeScorer([rouge_name], use_stemmer=True)



        self.scorer = FactualConsistencyScorer(align=set_align)



    def __str__(self):
        return f"ctc_fact_consistency"



    def _score(self, hypo, grounding):
        final_res = self.scorer.score(grounding, hypo)

        return final_res

    



    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
        white,
    ) -> np.ndarray:
        """
        Calculates Rouge score between stats['greedy_texts'] and target_texts.

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
                
                self._score(hyp, ref)
                for hyp, ref in zip(stats[greedy_text_key], target_texts)
            ]
        )
