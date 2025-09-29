import json
import os
from typing import List, Dict
import numpy as np
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from .utils import convert_to_json
from .evaluators import get_evaluator


class UniEval(GenerationMetric):
    """
    Calculates UniEval score between model-generated texts and ground truth texts.
    """

    def __init__(self, task='summarization', selected_key = 'overall', file_name ='./assist_res/unieval_res.json', depend=["greedy_texts"]):

        super().__init__(depend, "sequence")

        self.task = task # ['summarization', 'dialogue', 'fact']
        self.selected_key = selected_key # [coherence, consistency, fluency, relevance, overall]
        self.file_name = file_name
        if self.selected_key == 'overall':
            self.to_read_json_status = False
        else:
            self.to_read_json_status = True
        # remove ori_file
        if self.to_read_json_status == False:
            if os.path.isfile(self.file_name):
                os.remove(self.file_name)


    def __str__(self):
        return f"UniEval_{self.task}_{self.selected_key}"



    def _get_single_unieval_socre(self, pd_sent, gt_sent=None):
        src_list = [gt_sent] # should replace into the original text


        ref_list = [gt_sent]

        output_list = [pd_sent]

        # Prepare data for pre-trained evaluators
        if self.task == 'fact':
            data = convert_to_json(output_list=output_list,
                               src_list=src_list)
        else:
            data = convert_to_json(output_list=output_list,
                                   src_list=src_list, ref_list=ref_list)
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(self.task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, print_result=True)
        

        return eval_scores[0][self.selected_key]


    def list_to_json(self, pred_list, label_list, file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)

        res = {}
        res['pred'] = pred_list
        res['label'] = label_list


        with open(file_name, 'w') as f:
            json_str = json.dumps(res)
            f.write(json_str)
        print(f"{file_name} has saved all sampled predictions ['pred'] and labels ['label'] .")

    def json_to_list(self, file_name):
        with open(file_name, 'r') as ini_f:
            ini_str = ini_f.readline()
            res = json.loads(ini_str) # res['pred'], res['label']
        return res

    def read_res(self): # read a json file by row
        with open(self.file_name, 'r') as ini_f:
            final_res = {}
            for ini_str in ini_f:
                mid_res = json.loads(ini_str)  # res['pred'], res['label']
                final_res.update(mid_res)
        self.dict_res = final_res

    def mapping_res(self, hyp, ref, sou, key_word):
        cur_key = sou + ref
        return self.dict_res[cur_key][key_word]



    def _get_batch_unieval_socre(self, pd_sent, gt_sent=None, sr_sent=None):
        src_list = sr_sent # should replace into the original text


        ref_list = gt_sent

        output_list = pd_sent

        # Prepare data for pre-trained evaluators
        if self.task == 'fact':
            data = convert_to_json(output_list=output_list,
                               src_list=src_list)
        else:
            data = convert_to_json(output_list=output_list,
                                   src_list=src_list, ref_list=ref_list)
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(self.task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, print_result=True)


        if self.selected_key == 'overall':
            mid_res = []
            for i in range(len(eval_scores)):
                mid_val = eval_scores[i][self.selected_key]
                mid_res.append(mid_val)

            # save ori_file
            with open(self.file_name, 'a+') as f:
                for i in range(len(eval_scores)):
                    json_str = json.dumps(
                        # {(sr_sent[i] + gt_sent[i]): eval_scores[i]}
                        {sr_sent[i]: eval_scores[i]}
                    )
                    f.write(json_str)
                    f.write('\n')

        return mid_res


    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
        white=None,
    ) -> np.ndarray:
        """
        Calculates UniEval score between stats['greedy_texts'] and target_texts.

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

        if self.selected_key == 'overall':
            return np.array(self._get_batch_unieval_socre(stats[greedy_text_key], target_texts, stats["input_texts"]))
        else:
            if self.to_read_json_status == True:
                self.json_res = self.read_res()
            return np.array(
                [
                    # self._score_single(hyp, ref)
                    self.mapping_res(hyp, ref, sou, self.selected_key)
                    for hyp, ref, sou in zip(stats[greedy_text_key], target_texts, stats["input_texts"]) # prediction, ground truth, input
                ]
            )

