import numpy as np

from typing import List, Dict
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from .alignscore_modules.alignscore import AlignScore


class AlignScoreMetric(GenerationMetric):
    """
    Calculates SummaC score between model-generated texts and ground truth texts.
    """

    def __init__(
        self,
        model_card = 'roberta-base',
        ckpt_path = "/work/wataru-ha/decoding_uncertainty/alignscore_ckpt/AlignScore-large.ckpt",
        depend=["greedy_texts"]
    ):
        """
        Parameters:
            rouge_name (str): rouge metric type. Possible values:
                * rouge1
                * rouge2
                * rougeL

            model_card (str): the NLI model used for hallucination evaluation

        """
        super().__init__(depend, "sequence")

        self.model_card = model_card
        self.ckpt_path = ckpt_path
        self.scorer = AlignScore(
            model=self.model_card,
            batch_size=32,
            device="cuda",
            ckpt_path=self.ckpt_path,
            evaluation_mode='nli_sp',
        )

    def __str__(self):
        return f"factkb_{self.model_card}"

    def _get_align_score(
        self,
        context: str,
        answer: str,
    ):
        score = self.scorer.score(contexts=[context], claims=[answer])
        return float(score[0])
        
    @staticmethod
    def _prompt_replacer_for_summarization(
        text: str
    ) -> str:
        return text.replace(
            "# Instruction\nPlease summarize the following document.\n\n# Document: ",
            ""
        ).replace(
            "\n\n# Summary\n",
            ""
        )

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
            raise NotImplementedError()
        
        input_texts = stats["input_texts"]
        
        return np.array(
            [
                self._get_align_score(context, output)
                for output, context in zip(stats[greedy_text_key], input_texts)
            ]
        )
