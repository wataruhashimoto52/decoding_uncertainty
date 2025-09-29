
import os
import json
from typing import Optional

from huggingface_hub import login
from lm_polygraph.estimators import *
from estimators.rde import RDESeq as CustomRDESeq
from models import ITIWhiteboxModel, ContrastiveDecodingWhiteboxModel, FSDWhiteboxModel, FSDVecWhiteboxModel, SLEDWhiteboxModel
from utils.whitebox_model import WhiteboxModel, create_ensemble
from utils.dataset import Dataset
from lm_polygraph.utils.processor import Logger
from lm_polygraph.ue_metrics import PredictionRejectionArea, RiskCoverageCurveAUC
from metrics import SUMMACMetric, UniEval, P_K_Correlation, CTCFactConsistency, FactKBMetric, PassOneAccuracyMetric
from lm_polygraph.generation_metrics import RougeMetric, BartScoreSeqMetric, AccuracyMetric, AggregatedMetric, BLEUMetric, Comet, AlignScore
from utils.qa_manager import QAUEManager
from utils.manager import UEManager
from utils.savers import transfer_general_res_dict, save_list_dict_to_json
from utils.generation_parameters import GenerationParameters

from tap import Tap


UE_METHODS_MAPPER = {
    "information": "[MaximumSequenceProbability(), MeanTokenEntropy()]",  # information
    "density": "[MahalanobisDistanceSeq(\"decoder\"), CustomRDESeq(\"decoder\")]",  # density
    "sar": "[TokenSAR()]",  # [SAR(), SentenceSAR(),TokenSAR()]
    "se": "[SemanticEntropy()]",  # semantic entropy
    "ensemble": "[EPTtu(),EPTrmi(),PETtu(),PETrmi(),EPStu(),EPSrmi(),PEStu(),PESrmi(),EPSrmiabs(),PESrmiabs()]"  # ensembles
}

class Arguments(Tap):
    model_name_or_path: str
    dataset_name: str
    batch_size: int
    ue_methods_group: str
    device: str = "cuda:0"
    num_beams: int = 3
    use_small: int = 1
    use_rouge: int = 1
    use_bart: int = 1
    use_summac: int = 0
    use_ctc: int = 0
    use_factkb: int = 0
    use_spearmanr: int = 0
    use_kendalltau: int = 0
    use_bleu: int = 1
    use_comet: int = 1
    use_alignscore: int = 1
    use_accuracy: int = 1
    use_pass_one_accuracy: int = 1
    use_unieval_overall: int = 1
    num_beam_groups: int = 3
    diversity_penalty: float = 1.0
    penalty_alpha: float = 0.6
    top_k: int = 4
    dola_layers: str = "low"
    contrastive_decoding_alpha: float = 0.1
    contrastive_decoding_beta: float = 0.5
    fsd_topk: int = 5
    fsd_alpha: float = 0.6
    fsd_vec_topk: int = 5
    fsd_vec_alpha: float = 0.4
    sled_layers: str = "high"
    sled_mature_layer: int = 32
    sled_evolution_rate: float = 2.0
    sled_evolution_scale: int = 10
    sled_evolution_lower_bound: int = -100
    cd_amateur_model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    num_samples: Optional[int] = None
    repetition_penalty: float = 1.0
    decoding_strategy: str = "original"  # 'the name used to save the respective generation model'
    loogle_task_name: str = "qa"  # 'qa or summarization'
    temperature: Optional[float] = None
    
    
DATASET_MODEL_TO_PROMPT_PATHS = {
    ("trivia_qa", "meta-llama/Llama-2-7b-chat-hf"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "meta-llama/Llama-2-13b-chat-hf"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "OpenRLHF/Llama-3-8b-sft-mixture"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "OpenRLHF/Llama-3-8b-rlhf-100k"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "RLHFlow/LLaMA3-iterative-DPO-final"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "google/gemma-2-9b-it"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "meta-llama/Meta-Llama-3-8B"): "prompts/triviaqa_unaligned.txt",
    ("trivia_qa", "HuggingFaceH4/zephyr-7b-beta"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "Qwen/Qwen2.5-7B-Instruct"): "prompts/triviaqa_aligned.txt",
    ("trivia_qa", "Qwen/Qwen2.5-14B-Instruct"): "prompts/triviaqa_aligned.txt",
    # ("trivia_qa", "meta-llama/Meta-Llama-3-8B"): "prompts/triviaqa_aligned.txt",
    ("xsum", "OpenRLHF/Llama-3-8b-sft-mixture"): "prompts/xsum_aligned.txt",
    ("xsum", "OpenRLHF/Llama-3-8b-rlhf-100k"): "prompts/xsum_aligned.txt",
    ("xsum", "meta-llama/Llama-2-7b-chat-hf"): "prompts/xsum_aligned.txt",
    ("xsum", "RLHFlow/LLaMA3-iterative-DPO-final"): "prompts/xsum_aligned.txt",
    ("xsum", "meta-llama/Meta-Llama-3-8B"): "prompts/xsum_unaligned.txt",
    ("xsum", "meta-llama/Llama-2-13b-chat-hf"): "prompts/xsum_aligned.txt",
    ("xsum", "google/gemma-2-9b-it"): "prompts/xsum_aligned.txt",
    ("xsum", "HuggingFaceH4/zephyr-7b-beta"): "prompts/xsum_aligned.txt",
    ("xsum", "Qwen/Qwen2.5-7B-Instruct"): "prompts/xsum_aligned.txt",
    ("xsum", "Qwen/Qwen2.5-14B-Instruct"): "prompts/xsum_aligned.txt",
    ("humaneval", "meta-llama/Llama-2-7b-chat-hf"): "prompts/humaneval_aligned.txt",
    ("humaneval", "OpenRLHF/Llama-3-8b-sft-mixture"): "prompts/humaneval_aligned.txt",
    ("humaneval", "OpenRLHF/Llama-3-8b-rlhf-100k"): "prompts/humaneval_aligned.txt",
    ("humaneval", "RLHFlow/LLaMA3-iterative-DPO-final"): "prompts/humaneval_aligned.txt",
    ("humaneval", "meta-llama/Meta-Llama-3-8B"): "prompts/humaneval_unaligned.txt",
    ("humaneval", "meta-llama/Llama-2-13b-chat-hf"): "prompts/humaneval_aligned.txt",
    ("humaneval", "meta-llama/CodeLlama-7b-Instruct-hf"): "prompts/humaneval_aligned.txt",
    ("humaneval", "google/gemma-2-9b-it"): "prompts/humaneval_aligned.txt",
    ("humaneval", "HuggingFaceH4/zephyr-7b-beta"): "prompts/humaneval_aligned.txt",
    ("humaneval", "Qwen/Qwen2.5-7B-Instruct"): "prompts/humaneval_aligned.txt",
    ("humaneval", "Qwen/Qwen2.5-14B-Instruct"): "prompts/humaneval_aligned.txt",
    # ("google-research-datasets/mbpp", "meta-llama/Llama-2-7b-chat-hf"): "prompts/mbpp_aligned.txt",
    # ("google-research-datasets/mbpp", "meta-llama/Llama-2-7b"): "prompts/mbpp_unaligned.txt",
    ("wics/strategy-qa", "meta-llama/Llama-2-7b-chat-hf"): "prompts/strategyqa_aligned.txt",
    ("wics/strategy-qa", "meta-llama/Meta-Llama-3-8B"): "prompts/strategyqa_unaligned.txt",
    
    ("wmt19", "OpenRLHF/Llama-3-8b-sft-mixture"): "prompts/wmt_aligned.txt",
    ("wmt19", "OpenRLHF/Llama-3-8b-rlhf-100k"): "prompts/wmt_aligned.txt",
    ("wmt19", "meta-llama/Llama-2-7b-chat-hf"): "prompts/wmt_aligned.txt",
    ("wmt19", "RLHFlow/LLaMA3-iterative-DPO-final"): "prompts/wmt_aligned.txt",
    ("wmt19", "meta-llama/Meta-Llama-3-8B"): "prompts/wmt_unaligned.txt",
    ("wmt19", "meta-llama/Llama-2-13b-chat-hf"): "prompts/wmt_aligned.txt",
    ("wmt19", "google/gemma-2-9b-it"): "prompts/wmt_aligned.txt",
    ("wmt19", "HuggingFaceH4/zephyr-7b-beta"): "prompts/wmt_aligned.txt",
    ("wmt19", "Qwen/Qwen2.5-7B-Instruct"): "prompts/wmt_aligned.txt",
    ("wmt19", "Qwen/Qwen2.5-14B-Instruct"): "prompts/wmt_aligned.txt",
    # ("wmt19", "meta-llama/Meta-Llama-3-8B"): "prompts/wmt_aligned.txt",
    # ("loogle_qa", "togethercomputer/Llama-2-7B-32K-Instruct"): "prompts/loogle_qa_aligned.txt",
    # ("loogle_summarization", "togethercomputer/Llama-2-7B-32K-Instruct"): "prompts/loogle_summarization_aligned.txt",
}
    

def obtain_nlg_metric_list(args: Arguments):
    res = []
    if "trivia_qa" in args.dataset_name:
        if args.use_rouge:
            res.append('rouge')
    elif "xsum" in args.dataset_name:
        if args.use_rouge:
            res.append('rouge')
        if args.use_bart:
            res.append('bart')
        if args.use_factkb:
            res.append('factkb')
        if args.use_spearmanr:
            res.append('spearmanr')
        if args.use_kendalltau:
            res.append('kendalltau')
        if args.use_unieval_overall:
            res.append("unieval")
        if args.use_alignscore:
            res.append("alignscore")
    elif "strategy-qa" in args.dataset_name:
        if args.use_rouge:
            res.append('rouge')
    
    elif "wmt" in args.dataset_name:
        if args.use_bleu:
            res.append("bleu")
        if args.use_comet:
            res.append("comet")
        if args.use_alignscore:
            res.append("alignscore")
    
    elif "strategy-qa" in args.dataset_name:
        if args.use_accuracy:
            res.append('accuracy')
            
    elif "humaneval" in args.dataset_name:
        if args.use_pass_one_accuracy:
            res.append("pass_one_accuracy")
        
    return res


def get_nlg_metric(nlg_metric_list: list[str], assist_file_name: str, dataset_name: str):
    res = []
    depend = ["greedy_texts"]
    
    for ele in nlg_metric_list:
        if ele == 'rouge':
            if dataset_name == "trivia_qa":
                res.append(AggregatedMetric(RougeMetric("rougeL")))
            else:
                res.append(RougeMetric(rouge_name='rougeL'))
        elif ele == 'bart':
            res.append(BartScoreSeqMetric('rh'))
        elif ele == 'summac':
            res.append(SUMMACMetric(model_card='microsoft/deberta-base-mnli', depend=depend))
        elif ele == 'ctc':
            res.append(CTCFactConsistency(depend=depend, set_align='E-roberta'))
        elif ele == 'factkb':
            res.append(FactKBMetric())
        elif ele == 'spearmanr':
            res.append(P_K_Correlation(depend=depend, cor_type='spearmanr'))
        elif ele == 'kendalltau':
            res.append(P_K_Correlation(depend=depend, cor_type='kendalltau'))
        elif ele == 'unieval':
            res.append(UniEval(task='summarization', selected_key='overall', file_name=assist_file_name, depend=depend))
        elif ele == 'accuracy':
            res.append(AccuracyMetric())
        elif ele == "bleu":
            res.append(BLEUMetric())
        elif ele == "comet":
            res.append(Comet())
        elif ele == "pass_one_accuracy":
            res.append(PassOneAccuracyMetric())
        elif ele == "alignscore":
            res.append(AlignScore())

        else:
            raise ValueError(f'metric={ele} is wrongly set!')
        
    return res
    
    
def main(args: Arguments, seed: int, split_seed: int) -> None:
    args.as_dict()
    model_name_str = args.model_name_or_path.split("/")[-1]
    
    save_folder = f'./sample_res/{args.dataset_name}_{model_name_str}_{seed}/'
    assist_folder = f'./assist_res/{args.dataset_name}_{model_name_str}_{seed}/'
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    if not os.path.exists(assist_folder):
        os.mkdir(assist_folder)
        
    decoding_strategy_str = args.decoding_strategy
    
    generation_params_dict = {}
    
    if args.decoding_strategy == "beam_search":
        generation_params_dict = {
            "num_beams": args.num_beams,
        }
    elif args.decoding_strategy == "diverse_beam_search":
        generation_params_dict = {
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty,
        }
    elif args.decoding_strategy == "contrastive_search":
        generation_params_dict = {
            "penalty_alpha": args.penalty_alpha,
            "top_k": args.top_k,
        }
    elif args.decoding_strategy == "dola":
        generation_params_dict = {
            "dola_layers": args.dola_layers,
        }
    elif args.decoding_strategy == "contrastive_decoding":
        generation_params_dict = {
            "contrastive_decoding_alpha": args.contrastive_decoding_alpha,
            "contrastive_decoding_beta": args.contrastive_decoding_beta,
        }
    elif args.decoding_strategy == "fsd":
        generation_params_dict = {
            "fsd_topk": args.fsd_topk,
            "fsd_alpha": args.fsd_alpha,
        }
    elif args.decoding_strategy == "fsd_vec":
        generation_params_dict = {
            "fsd_vec_topk": args.fsd_vec_topk,
            "fsd_vec_alpha": args.fsd_vec_alpha,
        }
        
    elif args.decoding_strategy == "sled":
        generation_params_dict = {
            "sled_layers": args.sled_layers,
            "sled_mature_layer": args.sled_mature_layer,
            "sled_evolution_rate": args.sled_evolution_rate,
            "sled_evolution_scale": args.sled_evolution_scale,
            "sled_evolution_lower_bound": args.sled_evolution_lower_bound,
        }

    elif args.decoding_strategy == "original":
        if args.temperature is not None:
            generation_params_dict = {
                "temperature": args.temperature,
                "top_p": 1.0,
                "do_sample": True,
                "top_k": args.top_k,
            }
    
    else:
        raise NotImplementedError(f'decoding_strategy={args.decoding_strategy} is not implemented!')
    
    generation_params_dict.update(
        {"repetition_penalty": args.repetition_penalty}
    )

    generation_parameters = GenerationParameters(**generation_params_dict)
    
    decoding_strategy_str += "__" + "__".join([f"{k}={str(v)}" for k, v in generation_params_dict.items()])
    decoding_strategy_str += "_20250903"

    assist_file_name = os.path.join(assist_folder, f'{decoding_strategy_str}-unieval_res.json')
    ass_gt_file_name = os.path.join(assist_folder, f'{decoding_strategy_str}-unieval_gt.json')
    
    dataset_name_to_print = args.dataset_name.split("/")[-1]
    
    use_method_end = f"{dataset_name_to_print}_{args.ue_methods_group}_{decoding_strategy_str}"
                 
    print(use_method_end)
    print(f"model name: {args.model_name_or_path}")
    
    sample_save_file_name = os.path.join(save_folder, f'sample_{seed}-small_{args.use_small}-batchsize_{args.batch_size}-{use_method_end}.json')  # sample level generation metrics
    arg_params_save_file_name = os.path.join(save_folder, f"params_{seed}-small_{args.use_small}-batchsize_{args.batch_size}-{use_method_end}.json")
    general_save_file_name = os.path.join(save_folder, f'general_{seed}-small_{args.use_small}-batchsize_{args.batch_size}-{use_method_end}.json') # final results
    est_save_file_name = os.path.join(save_folder, f'est_{seed}-small_{args.use_small}-batchsize_{args.batch_size}-{use_method_end}.json') # estimation metrics self.estimations
    
    if args.use_small:
        if os.path.exists(sample_save_file_name):
            os.remove(sample_save_file_name)
        if os.path.exists(est_save_file_name):
            os.remove(est_save_file_name)
        if os.path.exists(general_save_file_name):
            os.remove(general_save_file_name)
        if os.path.exists(arg_params_save_file_name):
            os.remove(arg_params_save_file_name)

        if os.path.exists(assist_file_name):
            os.remove(assist_file_name)
        if os.path.exists(ass_gt_file_name):
            os.remove(ass_gt_file_name)
            
    if not args.use_small and os.path.exists(sample_save_file_name):
        raise ValueError(f'{sample_save_file_name} already exists!')
    if not args.use_small and os.path.exists(general_save_file_name):
        raise ValueError(f'{general_save_file_name} already exists!')
        
        
    if args.decoding_strategy == "iti":
        model = ITIWhiteboxModel.from_pretrained(
            args.model_name_or_path,
            device=args.device,
            parameters=generation_parameters,
        )
    elif args.decoding_strategy == "contrastive_decoding":
        model = ContrastiveDecodingWhiteboxModel.from_pretrained(
            args.model_name_or_path,
            args.cd_amateur_model_name_or_path,
            device=args.device,
            parameters=generation_parameters,
        )
        
    elif args.decoding_strategy == "fsd":
        model = FSDWhiteboxModel.from_pretrained(
            args.model_name_or_path,
            device=args.device,
            parameters=generation_parameters,
        )
        
    elif args.decoding_strategy == "fsd_vec":
        model = FSDVecWhiteboxModel.from_pretrained(
            args.model_name_or_path,
            device=args.device,
            parameters=generation_parameters,
        )
        
    elif args.decoding_strategy == "sled":
        model = SLEDWhiteboxModel.from_pretrained(
            args.model_name_or_path,
            device=args.device,
            parameters=generation_parameters,
        )
    else:
        model = WhiteboxModel.from_pretrained(
            args.model_name_or_path,
            device=args.device,
            parameters=generation_parameters,
        )
        
    model.set_generation_parameters(generation_parameters)

    # Train and Eval Datasets
    if 'trivia_qa' in args.dataset_name:
        question_col_name, answer_col_name = 'question', 'answer'
        with open(DATASET_MODEL_TO_PROMPT_PATHS[(args.dataset_name, args.model_name_or_path)], "r") as f:
            prompt = f.read()
        dataset = Dataset.load(
            (args.dataset_name, "rc.nocontext"),
            question_col_name,
            answer_col_name,
            batch_size=args.batch_size,
            prompt=prompt,
            split="validation",
            use_small=args.use_small,
            trust_remote_code=True,
        )

        train_dataset = Dataset.load(
            (args.dataset_name, "rc.nocontext"),
            question_col_name,
            answer_col_name,
            batch_size=args.batch_size,
            split="train",
            prompt=prompt,
            use_small=args.use_small,
            trust_remote_code=True
        )
    elif "xsum" in args.dataset_name:
        doc_col_name, summ_col_name = 'document', 'summary'
        with open(DATASET_MODEL_TO_PROMPT_PATHS[(args.dataset_name, args.model_name_or_path)], "r") as f:
            prompt = f.read()
        
        dataset = Dataset.load(
            args.dataset_name,
            doc_col_name,
            summ_col_name,
            batch_size=args.batch_size,
            prompt=prompt,
            split="test",
            use_small=args.use_small,
            trust_remote_code=True
        )
        dataset.select(list(range(0, 4000)))

        train_dataset = Dataset.load(
            args.dataset_name,
            summ_col_name, summ_col_name,
            batch_size=args.batch_size,
            split="train",
            prompt=prompt,
            use_small=args.use_small,
            trust_remote_code=True
        )
    elif "humaneval" in args.dataset_name:
        inputs, answer = 'prompt', 'test'
        with open(DATASET_MODEL_TO_PROMPT_PATHS[(args.dataset_name, args.model_name_or_path)], "r") as f:
            prompt = f.read()
            
        dataset = Dataset.load(
            "openai/openai_humaneval",
            inputs,
            answer,
            batch_size=args.batch_size,
            prompt=prompt,
            split="test",
            use_small=args.use_small,
            trust_remote_code=True
        )

        train_dataset = None
        
    elif "wmt" in args.dataset_name:
        source, target = 'de', 'en'
        with open(DATASET_MODEL_TO_PROMPT_PATHS[(args.dataset_name, args.model_name_or_path)], "r") as f:
            prompt = f.read()
        
        dataset = Dataset.load(
            ("wmt/wmt19", "de-en"),
            source,
            target,
            batch_size=args.batch_size,
            prompt=prompt,
            split="validation",
            use_small=args.use_small,
            trust_remote_code=True
        )
        
        # train_dataset = Dataset.load(
        #     (args.dataset_name, "de-en"),
        #     source,
        #     target,
        #     batch_size=args.batch_size,
        #     split="train",
        #     prompt=prompt,
        #     use_small=args.use_small,
        #     trust_remote_code=True
        # )
        train_dataset = None
        
    elif "strategy-qa" in args.dataset_name:
        inputs, answer = 'question', 'answerKey'
        with open(DATASET_MODEL_TO_PROMPT_PATHS[(args.dataset_name, args.model_name_or_path)], "r") as f:
            prompt = f.read()
            
        dataset = Dataset.load(
            args.dataset_name,
            inputs,
            answer,
            batch_size=args.batch_size,
            prompt=prompt,
            split="test",
            use_small=args.use_small,
            trust_remote_code=True
        )

        train_dataset = Dataset.load(
            args.dataset_name,
            source,
            target,
            batch_size=args.batch_size,
            split="train",
            prompt=prompt,
            use_small=args.use_small,
            trust_remote_code=True
        )
    elif "LooGLE" in args.dataset_name:
        if args.loogle_task_name == "qa":
            inputs, answer = 'question', 'answerKey'
        elif args.loogle_task_name == "summarization":
            doc_col_name, summ_col_name = 'context', 'answer'
        
        with open(
            DATASET_MODEL_TO_PROMPT_PATHS[(args.dataset_name, args.loogle_task_name, args.model_name_or_path)],
            "r" 
        ) as f:
            prompt = f.read()
            
        if args.loogle_task_name == "qa":
            dataset = Dataset.load(
                (args.dataset_name, "longdep_qa"),
                inputs,
                answer,
                batch_size=args.batch_size,
                prompt=prompt,
                split="test",
                use_small=args.use_small,
                trust_remote_code=True
            )
            train_dataset = None
        elif args.loogle_task_name == "summarization":
            dataset = Dataset.load(
                (args.dataset_name, "summarization"),
                doc_col_name,
                summ_col_name,
                batch_size=args.batch_size,
                prompt=prompt,
                split="test",
                use_small=args.use_small,
                trust_remote_code=True
            )
            train_dataset = None
            
    else:
        raise ValueError(f'dataset_name={args.dataset_name} is wrongly set!')
    
    if args.use_small:
        # 20
        dataset.subsample(20, seed=split_seed)
        # train_dataset.subsample(8, seed=split_seed)
    else:
        if args.num_samples:
            dataset.subsample(args.num_samples, seed=split_seed)
        # train_dataset.subsample(1000, seed=split_seed)
    
    
        
    nlg_metric_keywords = obtain_nlg_metric_list(args)
        
    ue_metrics = [
        # RiskCoverageCurveAUC(),
        PredictionRejectionArea(),
    ]
    metrics = get_nlg_metric(nlg_metric_keywords, assist_file_name, args.dataset_name)
    loggers = [Logger()]
    
    ue_methods = eval(UE_METHODS_MAPPER[args.ue_methods_group])
    
    # Initialize UE Manager
    ensemble_model = None
    
    if args.ue_methods_group == "ensemble":
        ensemble_model = create_ensemble(
            model_paths=[args.model_name_or_path],
            mc=True,
            seed=seed,
            mc_seeds=[2],
            ensembling_mode="pe",
            device="cuda:0",
            dropout_rate=0.1,
        )
    
    if args.dataset_name in ("trivia_qa", "strategy-qa"):
        man = QAUEManager(
            dataset,
            model,
            ue_methods,
            metrics,
            ue_metrics,
            loggers,
            train_data=train_dataset,
            sample_save_file_name=sample_save_file_name,
            est_save_file_name=est_save_file_name,
            ensemble_model=ensemble_model,
            ass_gt_file_name=ass_gt_file_name,
            # used for saving predictions
            # cal_save_embeddings=args.cal_save_embeddings,
            # result_dict_save_file_name=result_dict_save_file_name,
        )
    else:
        man = UEManager(
            dataset,
            model,
            ue_methods,
            metrics,
            ue_metrics,
            loggers,
            train_data=train_dataset,
            sample_save_file_name=sample_save_file_name,
            est_save_file_name=est_save_file_name,
            ensemble_model=ensemble_model,
            ass_gt_file_name=ass_gt_file_name,
            # used for saving predictions
            # cal_save_embeddings=args.cal_save_embeddings,
            # result_dict_save_file_name=result_dict_save_file_name,
        )

    # Compute Results
    results = man()
    
    for key in results.keys():
        print(key)
        print(f"UE Method: {key[1]}, NLG Metric: {key[2]}, UE Metric: {key[3]}, Final Score: {results[key]:.3f}")

    tranfer_general_results = transfer_general_res_dict(results)
    print(tranfer_general_results)
    if os.path.exists(general_save_file_name):
        os.remove(general_save_file_name)
    save_list_dict_to_json(d_dict=tranfer_general_results, file_name=general_save_file_name)

    print(f'{os.path.join(os.getcwd(), est_save_file_name)}')
    print(f'{os.path.join(os.getcwd(), sample_save_file_name)}')
    print(f'{os.path.join(os.getcwd(), assist_file_name)}')
    print(f'{os.path.join(os.getcwd(), general_save_file_name)}')
    
    with open(arg_params_save_file_name, "w") as f:
        d = args.as_dict()
        sorted_keys = sorted(d.keys())
        sorted_dict_by_key = {k: d[k] for k in sorted_keys}
        json.dump(sorted_dict_by_key, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    login(token=os.getenv("HF_TOKEN"))
    args = Arguments().parse_args()
    NUM_TRIALS = 1
    for seed in range(1, 1 + NUM_TRIALS):
        main(args, seed, seed)
