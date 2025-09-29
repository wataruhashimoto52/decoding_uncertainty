import re
import numpy as np

from typing import List, Dict
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


class PassOneAccuracyMetric(GenerationMetric):
    def __init__(
        self,
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
        self.pattern = re.compile(r"def\s+\w+\s*\([^)]*\):\n(?:\s+.*\n?)+")

    def __str__(self):
        return f"pass_one_accuracy"
    
    def run_tests(self, candidate_code: str, prompt: str, test_code: str) -> bool:
        """
        Executes the candidate code and then the test code in a common namespace.
        Returns True if execution of the test code does not raise an exception.
        """
        # Create a new namespace dictionary for exec()
        
        code_matcher = self.pattern.search(candidate_code)
        if not code_matcher:
            return False
        
        extracted_code = candidate_code[slice(*code_matcher.span())]
        raw_code = prompt + "\n" + candidate_code
        
        namespace = {}
        try:
            # Execute the candidate code (which should define the function)
            exec(extracted_code, namespace)
            # Execute the test code; tests use assertions and call the function by name.
            exec(test_code, namespace)
            return True
        except Exception as e:
            # Uncomment the next line to print the exception details for debugging
            print("Test failed with exception:", e)
            
            try:
                namespace = {}
                exec(raw_code, namespace)
                exec(test_code, namespace)
                return True
            except Exception as e:
                return False

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        white = True,
    ) -> np.ndarray:
        """
        Calculates SummaC score between model-generated texts and ground truth texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of Rouge Scores for each sample in input.
        """
        if white:
            greedy_text_key = "greedy_texts"
        else:
            raise NotImplementedError()
        
        input_texts = stats["input_texts"]
        
        # for generated, prompt, target in zip(stats["greedy_texts"], input_texts, target_texts):
        #     print("prompt: ", prompt)
        #     print("generated: ", generated)
        #     code_matcher = self.pattern.search(generated)
        #     if not code_matcher:
        #         extracted_code = ""
        #     else:
        #         extracted_code = generated[slice(*code_matcher.span())]
        #     print("extracted: ", extracted_code)
        #     print("test code: ", target)
            
        #     print("results: ", self.run_tests(generated, target))
        #     print()
            
        
        return np.array(
            [
                float(int(self.run_tests(generated, prompt, target)))
                for generated, prompt, target in zip(stats["greedy_texts"], input_texts, target_texts)
            ]
        )
