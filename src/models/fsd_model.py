import torch
import sys

from typing import List, Dict, Union
from abc import abstractmethod, ABC
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    BartForConditionalGeneration,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Gemma2ForCausalLM,
    MistralForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput
from dataclasses import asdict
from utils.generation_parameters import GenerationParameters
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from .fsds.ngram_model.fsd import NGram
from .fsds.ngram_model.fsd_vec import HiddenSoftNGram
from .fsds.utils import topk_logits_filter


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


class FSDWhiteboxModel(Model):
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
        
        generated_scores = []
        input_ids = args["input_ids"]
        prompt_len = args["attention_mask"].sum(dim=1)
        
        eos_token_id = self.tokenizer.eos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        eos_token_id_tensor = (
            torch.tensor([eos_token_id]).to(self.model.device)
            if eos_token_id is not None
            else None
        )
        
        batch_size, prefix_len = input_ids.size()
        unfinished_sequences = input_ids.new(batch_size).fill_(1)
        
        ng_list: list[NGram] = []
        for i, inputs in enumerate(input_ids):
            ng = NGram(
                input_ids=inputs.tolist()[prefix_len - prompt_len[i] :],
                n=3,
                vocab_size=self.model.config.vocab_size,
                beta=0.9,
                sw_coeff=0.0,
                stop_words_ids=[]
            )
            ng_list.append(ng)
            
        stopping_criteria = StoppingCriteriaList()
        
        with torch.inference_mode():
            generated_scores = ()
            generated_logits = ()
            
            for step in range(self.max_new_tokens):
                expert_lm_output = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )

                expert_logits = expert_lm_output.logits[:, -1, :]  # batch_size x vocab_size

                # avoid generating eos
                if eos_token_id != None:
                    expert_logits[:, eos_token_id] = -float("inf")
                
                next_token_probs = torch.nn.functional.softmax(expert_logits, dim=-1)
                next_token_probs = topk_logits_filter(next_token_probs, self.generation_parameters.fsd_topk)

                penalty_list = []
                for i, inputs in enumerate(input_ids):
                    _, b = torch.topk(next_token_probs[i], k=self.generation_parameters.fsd_topk)
                    penalty_i = ng_list[i].penalize(
                        inputs.tolist()[prefix_len - prompt_len[i] :], b.tolist()
                    )
                    penalty_list.append(penalty_i.view(1, -1))

                batch_penalty = torch.cat(penalty_list, dim=0)
                batch_penalty = batch_penalty.to(self.model.device)

                next_token_probs = (1 - self.generation_parameters.fsd_alpha) * next_token_probs \
                    - self.generation_parameters.fsd_alpha * batch_penalty
                    
                next_tokens = torch.argmax(next_token_probs, dim=-1) # batch_size
                
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                
                # update ngram model
                for i, token in enumerate(next_tokens):
                    ng_list[i].update(token.tolist())
                    
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                        .ne(eos_token_id_tensor.unsqueeze(1))
                        .prod(dim=0)
                    )
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids, None
                )
                
                generated_scores += (expert_logits.log_softmax(dim=-1), )
                generated_logits += (expert_logits, )
                
                if unfinished_sequences.max() == 0 or step == self.max_new_tokens - 1:
                    stopped = True
                else:
                    stopped = False

                if stopped:
                    break
                    
                if all([i == self.tokenizer.eos_token_id for i in next_tokens.tolist()]):
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
    def from_pretrained(model_path: str, device: str = "cpu", **kwargs):
        """
        Initializes the model from HuggingFace. Automatically determines model type.

        Parameters:
            model_path (str): model path in HuggingFace.
            device (str): device to load the model on.
        """
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
        if 'llama' in model_path.lower():
            model_type = "CausalLM"
            model = LlamaForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,)

            # model = model.bfloat16()
            print(model)
        elif 'mixtral' in model_path.lower() or "zephyr" in model_path.lower():
            model_type = "CausalLM"
            model = MistralForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,)

            # model = model.bfloat16()
            print(model)
            
        elif 'qwen2' in model_path.lower():
            model_type = "CausalLM"

            model = Qwen2ForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,)

            # model = model.bfloat16()
            print(model)

            
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

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_bos_token=True,
            model_max_length=1024,
            trust_remote_code=True,
            **kwargs,
        )

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return FSDWhiteboxModel(model, tokenizer, model_path, model_type)

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


class FSDVecWhiteboxModel(Model):
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
        
        generated_scores = []
        input_ids = args["input_ids"]
        
        eos_token_id = self.tokenizer.eos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        eos_token_id_tensor = (
            torch.tensor([eos_token_id]).to(self.model.device)
            if eos_token_id is not None
            else None
        )
        
        batch_size, _ = input_ids.size()
        unfinished_sequences = input_ids.new(batch_size).fill_(1)
        
        ng = HiddenSoftNGram(
            n=2,
            device=self.model.device,
            vocab_size=self.model.config.vocab_size,
            sw_coeff=0.0,
            stop_words_ids=[]
        )
        stopping_criteria = StoppingCriteriaList()
        
        with torch.inference_mode():
            generated_scores = ()
            generated_logits = ()
            
            for step in range(self.max_new_tokens):
                expert_lm_output = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )

                expert_logits = expert_lm_output.logits[:, -1, :]  # batch_size x vocab_size
                hidden_states = expert_lm_output.hidden_states[-1]

                # avoid generating eos
                if eos_token_id != None:
                    expert_logits[:, eos_token_id] = -float("inf")
                
                ng.update(hidden_states)
                
                next_token_probs = torch.nn.functional.softmax(expert_logits, dim=-1)
                next_token_probs = topk_logits_filter(next_token_probs, self.generation_parameters.fsd_vec_topk)

                batch_penalty = ng.penalize(input_ids, hidden_states.dtype)
                batch_penalty = batch_penalty.to(self.model.device)
                next_token_probs = (1 - self.generation_parameters.fsd_vec_alpha) * next_token_probs \
                    - self.generation_parameters.fsd_vec_alpha * batch_penalty

                next_tokens = torch.argmax(next_token_probs, dim=-1)
                # fsd-vec
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (
                    1 - unfinished_sequences
                )

                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                        .ne(eos_token_id_tensor.unsqueeze(1))
                        .prod(dim=0)
                    )
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids, None
                )
                generated_scores += (expert_logits.log_softmax(dim=-1), )
                generated_logits += (expert_logits, )
                
                if unfinished_sequences.max() == 0 or step == self.max_new_tokens - 1:
                    stopped = True
                else:
                    stopped = False

                if stopped:
                    break
                    
                if all([i == self.tokenizer.eos_token_id for i in next_tokens.tolist()]):
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
    def from_pretrained(model_path: str, device: str = "cpu", **kwargs):
        """
        Initializes the model from HuggingFace. Automatically determines model type.

        Parameters:
            model_path (str): model path in HuggingFace.
            device (str): device to load the model on.
        """
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
        if 'llama' in model_path.lower():
            model_type = "CausalLM"
            model = LlamaForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,)

            # model = model.bfloat16()
            print(model)
        
        elif 'mixtral' in model_path.lower() or "zephyr" in model_path.lower():
            model_type = "CausalLM"
            model = MistralForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,)
            print(model)
            
        elif 'qwen2' in model_path.lower():
            model_type = "CausalLM"

            model = Qwen2ForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,)

            # model = model.bfloat16()
            print(model)
        
            
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

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_bos_token=True,
            model_max_length=1024,
            trust_remote_code=True,
            **kwargs,
        )

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return FSDVecWhiteboxModel(model, tokenizer, model_path, model_type)

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

