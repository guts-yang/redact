import functools


def apply_embeddings_extraction_hook(module, embeddings):
    """Capture ``module``'s forward output into ``embeddings``.

    Attaches a forward hook that pushes a detached copy of the module's
    output onto the caller-owned ``embeddings`` list every time the
    module runs. Returns the :class:`torch.utils.hooks.RemovableHandle`
    so the caller can detach this specific hook later; each call
    registers an independent hook and yields a distinct handle.
    """

    def _record(_module, _inputs, output):
        embeddings.append(output.detach())

    return module.register_forward_hook(_record)


def remove_hooks(model):
    """Detach every forward hook from ``model`` and its submodules.

    Empties each module's ``_forward_hooks`` registry in place rather
    than rebinding it, so any external references to those mappings stay
    valid. Backward hooks are left alone, and modules with no forward
    hooks are simply skipped (the call is a no-op when nothing is
    registered).
    """
    for submodule in model.modules():
        forward_hooks = submodule._forward_hooks
        while forward_hooks:
            forward_hooks.popitem()


def get_nested_attr(obj, attr):
    """Follow a dotted attribute path, e.g. ``"model.embed_tokens"``.

    Resolves each ``.``-separated segment with :func:`getattr` in turn,
    returning the attribute at the end of the chain.
    """
    return functools.reduce(getattr, attr.split("."), obj)


# LRM-target chat-template wrappers used by ``RTOFU.prompt_formatted``.
# The labeled-data pipeline strips these via :func:`clean_question` so the
# leak classifier trains on bare prompts; the contextual-input builder
# applies the same strip at inference so the train/inference distributions
# stay aligned.
PROMPT_PREFIX = "<｜User｜>"
PROMPT_SUFFIX = "<｜Assistant｜>"


def clean_question(q: str) -> str:
    """Strip the LRM-target chat-template markers from a prompt.

    R-TOFU wraps each question in DeepSeek-R1-Distill's chat template
    (``<｜User｜>...<｜Assistant｜>``). The labeled-data pipeline stores
    the bare question in the ``prompt`` field, and the contextual leak
    classifier trains on that bare form. This helper is the single
    canonical strip — called both at labeling time (to populate the
    training data) and inside :func:`redact.regen.contextual_inputs.build_contextual_input`
    (to normalize whatever the engine passes at inference).

    Idempotent on already-bare strings.
    """
    if q.startswith(PROMPT_PREFIX):
        q = q[len(PROMPT_PREFIX):]
    if q.endswith(PROMPT_SUFFIX):
        q = q[:-len(PROMPT_SUFFIX)]
    return q.strip()


