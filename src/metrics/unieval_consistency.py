import os
import json
from typing import Dict, List

import numpy as np
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


class UniEval_Consistency(GenerationMetric):


    def __init__(self, task='summarization', selected_key = 'consistency', file_name ='./assist_res/unieval_res.json', depend=["greedy_texts"]):


        # super().__init__(task=task, selected_key =selected_key, file_name =file_name)
        super().__init__(depend, "sequence")


        self.task = task # ['summarization', 'dialogue', 'fact']
        self.selected_key = selected_key # [coherence, consistency, fluency, relevance, overall]
        self.file_name = file_name
        self.to_read_json_status = True

    def __str__(self):
        return f"UniEval_{self.task}_{self.selected_key}"






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

    def mapping_res(self, hyp, ref, key_word):
        cur_key = hyp + ref
        return self.dict_res[cur_key][key_word]






    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
        white=None,
    ) -> np.ndarray:




        if white:
            greedy_text_key = "greedy_texts"
        else:
            greedy_text_key = "blackbox_greedy_texts"
        if self.selected_key == 'overall':
            return np.array(self._get_batch_unieval_socre(stats[greedy_text_key], target_texts))
        else:
            if self.to_read_json_status == True:
                self.json_res = self.read_res()
            return np.array(
                [
                    # self._score_single(hyp, ref)
                    self.mapping_res(hyp, ref, self.selected_key)
                    for hyp, ref in zip(stats[greedy_text_key], target_texts)
                ]
            )
