import torch
import sys

from typing import List, Dict, Union
import numpy as np
from abc import abstractmethod, ABC
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    BartForConditionalGeneration,
    Qwen2ForCausalLM,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
import torch.nn.functional as F
from dataclasses import asdict
from utils.generation_parameters import GenerationParameters
from transformers.generation.utils import GenerateDecoderOnlyOutput


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


class SLEDWhiteboxModel(Model):
    max_new_tokens: int = 150
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
    
    def relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        scores_normalized[scores_normalized < probs_thresh] = filter_value
        return scores_normalized

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
        input_ids = args["input_ids"]
        args["output_hidden_states"] = True
        args["return_dict"] = True
        relative_top = 0.1
        base_layer = None
        mature_layer = self.generation_parameters.sled_mature_layer

        stopping_criteria = StoppingCriteriaList()
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        # init attention / hidden states / scores tuples
        scores = ()
        all_logits = ()
        decoder_attentions = ()
        decoder_hidden_states = ()

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        
        sled_layers = self.generation_parameters.sled_layers
        final_layer = self.model.config.get_text_config().num_hidden_layers
        # if the model has tied word embeddings, we skip the word embeddings (0-th) layer and start from the 2nd layer,
        # as the early exit from word embeddings will become identity function
        # if the model is really shallow (<=2 layers), we use the 1st layer if it's not the final layer and the 0-th
        # layer otherwise. Notice that DoLa does not help shallow models much.
        if not self.model.config.tie_word_embeddings:
            start_layer = 0
        elif final_layer > 2:
            start_layer = 2
        elif final_layer == 2:
            start_layer = 1
        else:
            start_layer = 0

        if isinstance(sled_layers, str) and sled_layers == "low":
            if start_layer == final_layer // 2:
                candidate_premature_layers = [start_layer]
            else:
                candidate_premature_layers = (
                    list(range(start_layer, final_layer // 2, 2))
                    if final_layer <= 40
                    else list(range(start_layer, 20, 2))
                )
        elif isinstance(sled_layers, str) and sled_layers == "high":
            candidate_premature_layers = (
                list(range(final_layer // 2, final_layer, 2))
                if final_layer <= 40
                else list(range(final_layer - 20, final_layer, 2))
            )
        # Set the `dola_layers` to a list of integers for layer indices to contrast manually specified layers.
        elif isinstance(sled_layers, list):
            candidate_premature_layers = [i for i in sled_layers if i < final_layer]
        elif isinstance(sled_layers, str) and sled_layers == "all":
            candidate_premature_layers = [i for i in range(final_layer) if i < mature_layer]
        else:
            raise ValueError("dola_layers must be either 'low', 'high' or a list of integers.")
        
        early_exit_layers = candidate_premature_layers + [mature_layer]
        
        
        for step in range(self.max_new_tokens):
        
            # forward pass to get next token
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            
            if early_exit_layers:
                dict_outputs = {}
                for i, early_exit_layer in enumerate(early_exit_layers):
                    logits = self.model.lm_head(outputs.hidden_states[early_exit_layer])
                    dict_outputs[early_exit_layer] = logits

            if base_layer is not None:
                base_logits = dict_outputs[base_layer][:, -1, :]
                final_logits = dict_outputs[mature_layer][:, -1, :]
                if relative_top > 0.0:
                    final_logits = self.relative_top_filter(final_logits, relative_top)
                    base_logits = base_logits.log_softmax(dim=-1)
                    mask = final_logits[0] < -1e3
                    base_logits[0][mask] = -1e3

                logits = final_logits - base_logits
                next_token_logits = logits
            else:
                new_output_logits = dict_outputs[mature_layer].clone()

                stacked_premature_layers = torch.stack([dict_outputs[i][:, -1, :] for i in candidate_premature_layers],
                                                        dim=0)
                softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, -1, :], dim=-1)
                softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)
                
                topk_prob, topk_indices = torch.topk(softmax_mature_layer, self.generation_parameters.sled_evolution_scale)
                topk_indices = topk_indices[0]

                divergence = stacked_premature_layers - dict_outputs[mature_layer][:, -1, :]
                
                batch_size = softmax_mature_layer.size(0)
                vocab_size = softmax_mature_layer.size(1)
                N_cl = softmax_premature_layers.size(0)
                
                # print(softmax_premature_layers.size())
                candidate_gradients_expanded = softmax_premature_layers.unsqueeze(2).expand(N_cl, batch_size, len(topk_indices), vocab_size)
                # candidate_gradients_expanded = softmax_premature_layers.expand(-1, len(topk_indices), self.model.config.vocab_size)
                candidate_mask = torch.zeros_like(candidate_gradients_expanded)
                # topk_indices_expanded = topk_indices.unsqueeze(0).unsqueeze(2)
                topk_indices_expanded = topk_indices.unsqueeze(0).unsqueeze(2).expand(
                    N_cl, batch_size, self.generation_parameters.sled_evolution_scale, 1
                )
                # candidate_mask.scatter_(2, topk_indices_expanded.expand(softmax_premature_layers.size(0), -1, -1), 1)
                candidate_mask.scatter_(3, topk_indices_expanded, 1)

                candidate_gradients_expanded = candidate_gradients_expanded - candidate_mask
                candidate_gradients_expanded = candidate_gradients_expanded.to(torch.float32)
                
                # layer_divergence_expanded = divergence.to(torch.float32)
                layer_divergence_expanded = divergence.unsqueeze(2).to(torch.float32)
                
                # layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded, dim=2)
                layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded, dim=3)
                layer_topk_values, layer_topk_indices = torch.topk(layer_dot_results, self.generation_parameters.sled_evolution_scale)
                # layer_topk_topk_indices = topk_indices[layer_topk_indices]
                
                # topk_indices: (batch_size, k) -> (N_cl, batch_size, k)
                topk_indices_expanded_for_gather = topk_indices.unsqueeze(0).expand(N_cl, batch_size, self.generation_parameters.sled_evolution_scale)
                # layer_topk_topk_indices: (N_cl, batch_size, k)
                layer_topk_topk_indices = torch.gather(topk_indices_expanded_for_gather, 2, layer_topk_indices)
                
                layer_topk_values = (layer_topk_values * (layer_topk_values > 0).to(layer_topk_values.dtype)) ** 2
                layer_topk_values_sum_layers = torch.sum(layer_topk_values, dim=2).clone()
                non_zero_indices = layer_topk_values_sum_layers != 0
                layer_topk_values[non_zero_indices] /= layer_topk_values_sum_layers[non_zero_indices].unsqueeze(1)
                if layer_topk_values_sum_layers.sum() != 0:
                    layer_topk_values_sum_layers = layer_topk_values_sum_layers / layer_topk_values_sum_layers.sum()

                candidate_delta = torch.zeros((N_cl, batch_size, vocab_size),
                                            device=softmax_mature_layer.device,
                                            dtype=softmax_mature_layer.dtype)
                for i in range(N_cl):
                    # layer_topk_topk_indices[i]: (batch_size, k)
                    # layer_topk_values[i]: (batch_size, k)
                    candidate_delta[i].scatter_(
                        1, layer_topk_topk_indices[i], -layer_topk_values[i].to(softmax_mature_layer.dtype)
                    )
                proxy_gradients_tensor_delta = torch.sum(candidate_delta * layer_topk_values_sum_layers.unsqueeze(2), dim=0)
                proxy_gradients_tensor_delta = proxy_gradients_tensor_delta.to(softmax_mature_layer.dtype)
                hidden_states_seq_i = new_output_logits[:, -1, :].clone()

                op_T = 1
                evolution_rate_scheduler = [
                    self.generation_parameters.sled_evolution_rate * (1 - i / op_T) for i in range(op_T)
                ]
                for op_t in range(op_T):
                    er_t = evolution_rate_scheduler[op_t]
                    # softmax_hidden_states_seq_i: (batch_size, vocab_size)
                    softmax_hidden_states_seq_i = F.softmax(hidden_states_seq_i, dim=-1)
                    # proxy_gradients_tensor: (batch_size, vocab_size)
                    proxy_gradients_tensor = softmax_hidden_states_seq_i + proxy_gradients_tensor_delta
                    hidden_states_seq_i.sub_(er_t * proxy_gradients_tensor)

                # lower bound を適用した新たなロジットを生成:
                # hidden_states_seq_i_new: (batch_size, vocab_size)
                hidden_states_seq_i_new = torch.full_like(
                    hidden_states_seq_i,
                    fill_value=self.generation_parameters.sled_evolution_lower_bound,
                    device=hidden_states_seq_i.device,
                    dtype=hidden_states_seq_i.dtype
                )
                
                # バッチ内の各行に対するインデックス: shape (batch_size, 1)
                batch_indices = torch.arange(batch_size, device=hidden_states_seq_i_new.device).unsqueeze(1).expand(batch_size, self.generation_parameters.sled_evolution_scale)

                # 各バッチの topk_indices の位置に、hidden_states_seq_i の対応する値を代入
                hidden_states_seq_i_new[batch_indices, topk_indices] = hidden_states_seq_i.gather(1, topk_indices.unsqueeze(0))
                
                # new_output_logits の最新トークン位置を更新
                new_output_logits[:, -1, :] = hidden_states_seq_i_new
                next_token_logits = new_output_logits[:, -1, :]

            
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            scores += (next_tokens_scores,)
            all_logits += (next_token_logits,)
            decoder_hidden_states += (outputs.hidden_states,)
                            
            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True
                    

            if all(stopping_criteria(input_ids, scores).tolist()):
                this_peer_finished = True
                
            if this_peer_finished:
                break

        generation = GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=all_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
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
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map={"": 0},
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

            # model = model.bfloat16()
            print(model)
            
        elif "qwen2" in model_path.lower():
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

        return SLEDWhiteboxModel(model, tokenizer, model_path, model_type)

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
