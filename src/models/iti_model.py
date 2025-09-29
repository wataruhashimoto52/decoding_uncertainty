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
)

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


class ITIWhiteboxModel(Model):
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

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_path: str,
        model_type: str,
        parameters: GenerationParameters = GenerationParameters(),
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
        
        generation = self.model.generate(**args)

        orig_scores = [s.log_softmax(-1) for s in processor.scores]

        # override generation.scores with original scores from model
        generation.generation_scores = generation.scores
        generation.scores = orig_scores

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
        if 'llama' in model_path:
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

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
            
        if "honest_llama" in model_path:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                padding_side="left",
                add_bos_token=True,
                model_max_length=1024,
                **kwargs,
            )
        else:
            raise NotImplementedError()

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return ITIWhiteboxModel(model, tokenizer, model_path, model_type)

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


def create_ensemble(
    model_paths: List[str] = [],
    mc: bool = False,
    seed: int = 1,
    mc_seeds: List[int] = [1],
    ensembling_mode: str = "pe",
    device: str = "cpu",
    dropout_rate: float = 0.1,
    **kwargs,
) -> ITIWhiteboxModel:
    model = ITIWhiteboxModel.from_pretrained(model_paths[0], **kwargs)
    ens = model.model

    ens.__class__ = type(
        "EnsembleModel", (model.model.__class__, EnsembleGenerationMixin), {}
    )

    if mc:
        ens.mc = True
        ens.mc_seeds = mc_seeds
        ens.base_seed = seed
        ens.ensembling_mode = ensembling_mode
        ens.mc_models_num = len(mc_seeds)
        ens.mc_seeds = mc_seeds

        replace_dropout(
            ens.config._name_or_path, ens, p=dropout_rate, share_across_tokens=True
        )

        ens.to(device)
        ens.train()
    else:
        raise ValueError(
            "Only Monte-Carlo ensembling is available. Please set the corresponding argument value to True"
        )

    return model
