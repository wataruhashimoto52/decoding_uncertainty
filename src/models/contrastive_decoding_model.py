import torch
import sys
import copy

from typing import List, Dict, Union
from abc import abstractmethod, ABC
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    BartForConditionalGeneration,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput
from dataclasses import asdict
from utils.generation_parameters import GenerationParameters
from utils.ensemble_generator import EnsembleGenerationMixin
from utils.dropout import replace_dropout


def _validate_args(args):
    if "presence_penalty" in args.keys() and args["presence_penalty"] != 0.0:
        sys.stderr.write(
            "Skipping requested argument presence_penalty={}".format(
                args["presence_penalty"]
            )
        )

    # remove arguments that are not supported by the HF model.generate function
    keys_to_remove = ["presence_penalty", "generate_until", "allow_newlines"]
    for key in keys_to_remove:
        args.pop(key, None)

    return args



class Model(ABC):
    """
    Abstract model class. Used as base class for both White-box models and Black-box models.
    """

    def __init__(self, model_path: str, model_type: str):
        """
        Parameters:
            model_path (str): unique model path where it can be found.
            model_type (str): description of additional model properties. Can be 'Blackbox' or model specifications
                in the case of white-box.
        """
        self.model_path = model_path
        self.model_type = model_type

    @abstractmethod
    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Abstract method. Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def generate(self, **args):
        """
        Abstract method. Generates the model output with scores from batch formed by HF Tokenizer.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, **args):
        """
        Abstract method. Calls the model on the input batch. Returns the resulted scores.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")


class ContrastiveDecodingWhiteboxModel(Model):
    """
    White-box model class. Have access to model scores and logits. Currently implemented only for Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph import WhiteboxModel
    >>> model = WhiteboxModel.from_pretrained(
    ...     "bigscience/bloomz-3b",
    ...     device="cuda:0",
    ... )
    ```
    """
    max_new_tokens = 100
    attn_mode = "flash"

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        amateur_model: AutoModelForCausalLM,
        amateur_tokenizer: AutoTokenizer,
        model_path: str,
        model_type: str,
        parameters: GenerationParameters = GenerationParameters()
    ):
        """
        Parameters:
            model (AutoModelForCausalLM): HuggingFace model.
            tokenizer (AutoTokenizer): HuggingFace tokenizer.
            model_path (Optional[str]): Unique model path in HuggingFace.
            model_type (str): Additional model specifications.
            parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
        """
        super().__init__(model_path, model_type)
        self.model = model
        self.tokenizer = tokenizer
        self.amateur_model = amateur_model
        self.amateur_tokenizer = amateur_tokenizer
        self.generation_parameters = parameters
        
    def set_generation_parameters(self, generation_parameters: GenerationParameters) -> None:
        self.generation_parameters = generation_parameters

    class _ScoresProcessor:
        # Stores original token scores instead of the ones modified with generation parameters
        def __init__(self):
            self.scores = []

        def __call__(self, input_ids=None, scores=None):
            self.scores.append(scores.log_softmax(-1))
            return scores

    def generate(self, **args):
        """
        Generates the model output with scores from batch formed by HF Tokenizer.

        Parameters:
            **args: Any arguments that can be passed to model.generate function from HuggingFace.
        Returns:
            ModelOutput: HuggingFace generation output with scores overriden with original probabilities.
        """
        default_params = asdict(self.generation_parameters)

        # add ScoresProcessor to collect original scores
        processor = self._ScoresProcessor()
        if "logits_processor" in args.keys():
            logits_processor = LogitsProcessorList(
                [processor, args["logits_processor"]]
            )
        else:
            logits_processor = LogitsProcessorList([processor])
        args["logits_processor"] = logits_processor
        
        default_params.update(args)
        args = default_params
        args = _validate_args(args)
        args["output_scores"] = True
        args["output_logits"] = True
        args["return_dict_in_generate"] = True
        
        input_ids = args["input_ids"]
        
        generated_ids = []
        generated_scores = []
        
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.inference_mode():            
            generated_scores = ()
            generated_logits = ()
            last_input_token = args["input_ids"][:, -1]
            
            for _ in range(self.max_new_tokens):
                expert_lm_output = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )
                amateur_lm_output = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
                
                expert_logits = expert_lm_output.logits[:, -1, :]
                amateur_logits = amateur_lm_output.logits[:, -1, :]

                cutoff = (
                    torch.log(torch.tensor(
                        self.generation_parameters.contrastive_decoding_alpha,
                        device=expert_logits.device,
                    ))
                    + expert_logits.max(dim=-1, keepdim=True).values
                )
                
                diffs = (1 + self.generation_parameters.contrastive_decoding_beta) * expert_logits \
                    - self.generation_parameters.contrastive_decoding_beta * amateur_logits
                
                cdlogits_for_score = diffs.masked_fill(expert_logits < cutoff, -8.0)  # [B, V] 
                cdlogits = diffs.masked_fill(expert_logits < cutoff, -float("inf"))  # [B, V]
            
                if eos_token_id != None:
                    cdlogits[:, eos_token_id] = -float("inf")
            
                last_input_token = torch.argmax(cdlogits, dim=-1)
                # 要検討: 語彙の制限をスコア計算に含めるべきか，それとも制限前のlogitをもとに計算すべきか
                # next_token_scores = cdlogits_for_score.log_softmax(dim=-1)
                next_token_scores = diffs.log_softmax(dim=-1)
                
                input_ids = torch.cat([input_ids, last_input_token[:, None]], dim=-1)
                
                generated_scores += (next_token_scores, )
                generated_logits += (cdlogits, )

                if all([i == self.tokenizer.eos_token_id for i in last_input_token.tolist()]):
                    break
        
        sequences = input_ids
        logits = torch.stack(generated_logits)
        
        generation = GenerateDecoderOnlyOutput(
            sequences=sequences,
            scores=generated_scores,
            logits=logits,
            hidden_states=expert_lm_output.hidden_states,
            past_key_values=None,
        )

        return generation

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        args = _validate_args(args)
        args["return_dict_in_generate"] = True
        
        batch: Dict[str, torch.Tensor] = self.tokenize(input_texts)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        
        sequences = self.model.generate(**batch, **args).sequences.cpu()
        input_len = batch["input_ids"].shape[1]
        texts = []
        for seq in sequences:
            if self.model_type == "CausalLM":
                texts.append(self.tokenizer.decode(seq[input_len:]))
            else:
                texts.append(self.tokenizer.decode(seq[1:]))
        return texts

    def __call__(self, **args):
        """
        Calls the model on the input batch. Returns the resulted scores.
        """
        return self.model(**args)

    def device(self):
        """
        Returns the device the model is currently loaded on.

        Returns:
            str: device string.
        """
        return self.model.device

    @staticmethod
    def from_pretrained(model_path: str, amateur_model_path: str, device: str = "cpu", **kwargs):
        """
        Initializes the model from HuggingFace. Automatically determines model type.

        Parameters:
            model_path (str): model path in HuggingFace.
            device (str): device to load the model on.
        """
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
        amateur_model = None
        amateur_tokenizer = None
        
        if 'llama' in model_path.lower():
            model_type = "CausalLM"
            model = LlamaForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,)

            # model = model.bfloat16()
            print(model)
            
            amateur_model = LlamaForCausalLM.from_pretrained(
                amateur_model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )
        
        elif 'qwen2' in model_path.lower():
            model_type = "CausalLM"
            model = Qwen2ForCausalLM.from_pretrained(
                model_path,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
            ).to(device)
            print(model)
            amateur_model = Qwen2ForCausalLM.from_pretrained(
                amateur_model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            ).to(device)
            
            
        elif any(["CausalLM" in architecture for architecture in config.architectures]):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, max_length=256, trust_remote_code=True, **kwargs
            ).to(device)
        elif any(
            [
                ("Seq2SeqLM" in architecture)
                or ("ConditionalGeneration" in architecture)
                for architecture in config.architectures
            ]
        ):
            model_type = "Seq2SeqLM"
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, max_length=1024, **kwargs
            ).to(device)
            if "falcon" in model_path:
                model.transformer.alibi = True
        elif any(
            ["BartModel" in architecture for architecture in config.architectures]
        ):
            model_type = "Seq2SeqLM"
            model = BartForConditionalGeneration.from_pretrained(
                model_path, max_length=1024, **kwargs
            ).to(device)
        else:
            raise ValueError(
                f"Model {model_path} is not adapted for the sequence generation task"
            )
        if not kwargs.get("load_in_8bit", False) and not kwargs.get(
            "load_in_4bit", False
        ):
            model = model.to(device)
            if amateur_model:
                amateur_model = amateur_model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_bos_token=True,
            model_max_length=1024,
            trust_remote_code=True,
            **kwargs,
        )
        amateur_tokenizer = AutoTokenizer.from_pretrained(
            amateur_model_path,
            padding_side="left",
            add_bos_token=True,
            model_max_length=1024,
            trust_remote_code=True,
            **kwargs,
        )

        model.eval()
        if amateur_model:
            amateur_model.eval()
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return ContrastiveDecodingWhiteboxModel(
            model,
            tokenizer,
            amateur_model,
            amateur_tokenizer,
            model_path,
            model_type
        )

    def tokenize(
        self, texts: Union[List[str], List[List[Dict[str, str]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input texts batch into a dictionary using the model tokenizer.

        Parameters:
            texts (List[str]): list of input texts batch.
        Returns:
            dict[str, torch.Tensor]: tensors dictionary obtained by tokenizing input texts batch.
        """
        # Apply chat template if tokenizer has it
        if self.tokenizer.chat_template is not None:
            formatted_texts = []
            for chat in texts:
                if isinstance(chat, str):
                    chat = [{"role": "user", "content": chat}]
                formatted_chat = self.tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False
                )
                formatted_texts.append(formatted_chat)
            texts = formatted_texts

        return self.tokenizer(texts, padding=True, return_tensors="pt")
    
    def amateur_tokenize(
        self, texts: Union[List[str], List[List[Dict[str, str]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input texts batch into a dictionary using the model tokenizer.

        Parameters:
            texts (List[str]): list of input texts batch.
        Returns:
            dict[str, torch.Tensor]: tensors dictionary obtained by tokenizing input texts batch.
        """
        # Apply chat template if tokenizer has it
        if self.tokenizer.chat_template is not None:
            formatted_texts = []
            for chat in texts:
                if isinstance(chat, str):
                    chat = [{"role": "user", "content": chat}]
                formatted_chat = self.tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False
                )
                formatted_texts.append(formatted_chat)
            texts = formatted_texts

        return self.amateur_tokenizer(texts, padding=True, return_tensors="pt")
