import numpy as np
# from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from typing import List, Dict
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


class FactKBMetric(GenerationMetric):
    """
    Calculates SummaC score between model-generated texts and ground truth texts.
    """

    def __init__(
        self,
        tokenizer_card = 'roberta-base',
        model_card = 'bunsenfeng/FactKB',
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

        self.tokenizer_card = tokenizer_card
        self.model_card = model_card
        self.sent_toknizer = AutoTokenizer.from_pretrained(self.tokenizer_card)
        self.factkb_model = AutoModelForSequenceClassification.from_pretrained(model_card, num_labels=2).eval()
        self.factkb_model = self.factkb_model.cuda()

    def __str__(self):
        return f"factkb_{self.model_card}"

    def _get_fact_score(
        self,
        tokenizer,
        document,
        summary,
    ):
        inputs = [[summary, self._prompt_replacer_for_summarization(document)]]
        tokens = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True).to(self.factkb_model.device)
        result = torch.softmax(self.factkb_model(**tokens).logits, dim = 1)
        return float(result[0][1])
    
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
                self._get_fact_score(self.sent_toknizer, document, summary)
                for summary, document in zip(stats[greedy_text_key], input_texts)
            ]
        )
