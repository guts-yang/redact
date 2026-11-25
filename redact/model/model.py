import torch
from tokenizers import Regex
from tokenizers.decoders import ByteLevel as DecoderByteLevel
from tokenizers.pre_tokenizers import (
    ByteLevel as PreTokByteLevel,
    Metaspace,
    Sequence as PreTokSequence,
    Split,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from redact.utils import load_yaml


# GPT-4 / tiktoken pre-tokenization regex used by DeepSeek-R1-Distill and
# other byte-level BPE Llama-3 descendants. Lifted from the canonical
# tokenizer.json that ships with deepseek-ai/DeepSeek-R1-Distill-Llama-8B.
_GPT4_PRE_TOK_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _maybe_repair_byte_level_tokenizer(tokenizer):
    """Restore byte-level BPE pre_tokenizer/decoder for tokenizers that ship
    a broken Metaspace config.

    Symptom: transformers >=5.0 mutates the loaded backend tokenizer
    config for byte-level BPE Llama-3 descendants (e.g.
    DeepSeek-R1-Distill-Llama-8B) — the pre_tokenizer becomes
    ``Metaspace(replacement="▁", split=False)`` and the decoder becomes
    a Metaspace+ByteFallback Sequence, but the *vocabulary* still uses
    GPT-2-style ``Ġ`` byte markers. Result: encode strips spaces and
    produces fragmented tokens; decode leaves literal ``Ġ`` in the
    output. The on-disk ``tokenizer.json`` is unchanged and parses
    correctly under the bare ``tokenizers`` library; the override
    happens in transformers' ``LlamaTokenizerFast`` initialization.
    transformers 4.43 (R-TOFU's pinned version) leaves the expected
    ``Sequence(Split, ByteLevel)`` pre_tokenizer and ``ByteLevel``
    decoder in place, which is what we restore here.

    Detection: pre_tokenizer is Metaspace AND vocab contains a
    Ġ-prefixed token (proves the BPE merges expect byte-level inputs).
    Skipped when the pre_tokenizer is anything else (legitimate
    SentencePiece-only models keep their config) or when no Ġ-prefixed
    token exists (legitimate Metaspace-only vocab).

    Other call sites that load LRM-target's tokenizer outside HFModel
    (and therefore must call this helper themselves) include
    ``scripts/evaluate_rtofu_from_csv.py`` (TokenEntropy hazard) and the
    R2MU baseline at ``scripts/r2mu/unlearn.py``.
    """
    backend = tokenizer.backend_tokenizer
    if not isinstance(backend.pre_tokenizer, Metaspace):
        return False
    g_hello_id = tokenizer.convert_tokens_to_ids("ĠHello")
    if g_hello_id is None or g_hello_id == tokenizer.unk_token_id:
        return False

    backend.pre_tokenizer = PreTokSequence([
        Split(
            pattern=Regex(_GPT4_PRE_TOK_REGEX),
            behavior="isolated",
            invert=False,
        ),
        PreTokByteLevel(
            add_prefix_space=False, trim_offsets=True, use_regex=False,
        ),
    ])
    backend.decoder = DecoderByteLevel(
        add_prefix_space=True, trim_offsets=True, use_regex=True,
    )
    return True


# Model-name fragments whose remote code we decline to execute; everything
# else defaults to trusting the repo's custom modeling files.
_UNTRUSTED_NAME_FRAGMENTS = ("c4ai-command-r-v01", "falcon", "phi-1_5")

# When the loaded model lacks its own tokenizer artifacts, fall back to this
# reference vocabulary (keyed by a name fragment).
_TOKENIZER_FALLBACKS = (("openelm", "meta-llama/Llama-2-7b-hf"),)


def _should_trust_remote_code(model_name):
    """Decline custom remote code for a short denylist of model families."""
    lowered = model_name.lower()
    return not any(frag in lowered for frag in _UNTRUSTED_NAME_FRAGMENTS)


def _tokenizer_source(model_name, hf_name):
    """Pick which checkpoint name supplies the tokenizer."""
    lowered = model_name.lower()
    for fragment, fallback in _TOKENIZER_FALLBACKS:
        if fragment in lowered:
            return fallback
    return hf_name


class HFModel:
    """Thin loader around a HuggingFace causal LM and its tokenizer.

    Reads a per-model YAML config (``{config_path}/{model_name}.yaml``)
    describing the checkpoint name, attention backend, and quantization
    flags, then materializes the model and tokenizer. ``__call__`` and
    ``generate`` forward to the underlying HF model after dropping a few
    caller-supplied kwargs that certain architectures reject.
    """

    def __init__(
        self,
        model_name,
        model_path=None,
        config_path="./config",
        generation_config=None,
    ):
        self.model_name = model_name
        self.model_config = load_yaml(f"{config_path}/{model_name}.yaml")

        want_4bit = self.model_config["load_in_4bit"]
        want_8bit = self.model_config["load_in_8bit"]
        quantization_config = None
        if want_4bit or want_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=want_4bit,
                load_in_8bit=want_8bit,
            )

        trust_remote_code = _should_trust_remote_code(model_name)
        checkpoint = model_path or self.model_config["hf_name"]
        dtype_name = str(self.model_config.get("torch_dtype", "bfloat16")).lower()
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype in model config: {dtype_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype_map[dtype_name],
            attn_implementation=self.model_config["attn_implementation"],
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )

        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        tokenizer_source = (
            self.model_config.get("tokenizer_name")
            or _tokenizer_source(model_name, self.model_config["hf_name"])
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=trust_remote_code,
        )
        # Net-new (OURS): undo the byte-level BPE tokenizer regression on
        # transformers >=5.0 for Llama-3 descendants. Leave untouched.
        if _maybe_repair_byte_level_tokenizer(self.tokenizer):
            print(
                f"Repaired byte-level BPE tokenizer config for {model_name} "
                f"(transformers regression workaround)"
            )

        if generation_config is None:
            generation_config = GenerationConfig(do_sample=False, use_cache=True)
        self.model.generation_config = generation_config
        self.generation_config = self.model.generation_config
        self.device = self.model.device

        # Reuse EOS as the pad token when the tokenizer ships none.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Qwen / StarCoder2 batched generation requires left padding.
        lowered = model_name.lower()
        if "qwen" in lowered or "starcoder2" in lowered:
            self.tokenizer.padding_side = "left"

    def _drop_token_type_ids_if_needed(self, kwargs, families):
        """Pop ``token_type_ids`` for architectures that reject it."""
        lowered = self.model_name.lower()
        rejects = any(name in lowered for name in families)
        if not rejects and self.model_name == "falcon-180B-chat":
            rejects = True
        if rejects:
            kwargs.pop("token_type_ids", None)

    def __call__(self, *args, **kwargs):
        # The eval engine threads ``prompts``/``answers`` through for
        # bookkeeping; the underlying model does not accept them.
        kwargs.pop("prompts", None)
        kwargs.pop("answers", None)
        self._drop_token_type_ids_if_needed(kwargs, ("olmo", "qwen"))
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        self._drop_token_type_ids_if_needed(kwargs, ("olmo",))
        return self.model.generate(*args, **kwargs)
