from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationParameters:
    """
    Parameters to override in model generation.

    Parameters:
        temperature (float): Temperature in sampling generation. Has no effect when `do_sample` is not set.
            Default: 1.0.
        top_k (int): Top-k token predictions to consider in sampling generation. Has no effect when `do_sample` is
            not set. Default: 1.
        top_p (float): Only consider the highest unique tokens, which probabilities sum up to `topp`. Has no effect
            when `do_sample` is not set. Default: 1.0.
        do_sample (bool): If true, perform sampling from models probabilities. If false, only generate token with
            maximum probability. Default: False.
        num_beams (int): Number of beams if beam search generation is used. Has no effect when `do_sample` is not
            set. Default: 1.
        presence_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
            they appear in the text so far, increasing the model's likelihood to talk about new topics. Applied for
            OpenAI-API blackbox models. Default: 0.0.
        repetition_penalty (float): The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no
            penalty. Applied for whitebox models from HuggingFace. Default: 1.0.
        allow_newlines (bool): If set, the model is not allowed to generate tokens with newlines. Default: False.
    """

    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    penalty_alpha: Optional[float] = None
    diversity_penalty: float = 0.0
    dola_layers: Optional[str] = None
    contrastive_decoding_alpha: Optional[float] = None
    contrastive_decoding_beta: Optional[float] = None
    fsd_topk: Optional[int] = None
    fsd_alpha: Optional[float] = None
    fsd_vec_topk: Optional[int] = None
    fsd_vec_alpha: Optional[float] = None
    sled_mature_layer: Optional[int] = None
    sled_layers: Optional[str] = None
    sled_evolution_rate: Optional[float] = None
    sled_evolution_scale: Optional[int] = None
    sled_evolution_lower_bound: Optional[float] = None
    repetition_penalty: float = 1.0
    allow_newlines: bool = True
