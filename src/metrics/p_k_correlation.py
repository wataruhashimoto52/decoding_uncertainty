import numpy as np
from sentence_transformers import SentenceTransformer

from typing import List, Dict
from scipy import stats

from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


class P_K_Correlation(GenerationMetric):
    def __init__(self, depend=["greedy_texts"], cor_type=''):
        super().__init__(depend, "sequence")
        
        self.sbert = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
        assert cor_type in ['spearmanr', 'kendalltau']
        self.cor_type = cor_type

    def __str__(self):
        return f"correlation_{self.cor_type}"

    

    def _cal_single_corr(
        self,
        gen_text,
        ref_text
    ) -> np.ndarray:
        hypo_embedding = self.sbert.encode(gen_text)
        ref_embedding = self.sbert.encode(ref_text)
        if self.cor_type == 'spearmanr':
            res = stats.spearmanr(hypo_embedding, ref_embedding)
        elif self.cor_type == 'kendalltau':
            res = stats.kendalltau(hypo_embedding, ref_embedding)
        else:
            raise ValueError(f'self.cor_type={self.cor_type} is wrongly set!')
        return res.statistic
        # return util.pairwise_cos_sim(embeddings, references).numpy()


    def __call__(
        self,
        prov_stats: Dict[str, np.ndarray],
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
                # self._score_single(hyp, ref)
                self._cal_single_corr(hyp, ref)
                for hyp, ref in zip(prov_stats[greedy_text_key], target_texts)
            ]
        )
