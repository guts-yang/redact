"""
Tests for transformers 5.x compatibility fixes:

1. Handle-based hook removal in RegeneratingModel (replaces _forward_hooks assignment)
2. _remove_hooks dispatcher in inference.py
"""

import torch
import torch.nn as nn

from redact.regen.utils import (
    apply_embeddings_extraction_hook,
    remove_hooks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleEmbedding(nn.Module):
    """Minimal embedding module to attach hooks to."""

    def __init__(self, dim=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(100, dim))

    def forward(self, x):
        return x


class FakeInnerModel(nn.Module):
    """Model with a nested embedding, mimicking model.model.embed_tokens."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = SimpleEmbedding()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(self.embed_tokens(x))


# ---------------------------------------------------------------------------
# Issue #3: Handle-based hook removal
# ---------------------------------------------------------------------------


class TestRemoveHooks:
    """Tests for the standalone remove_hooks function in redact/regen/utils.py."""

    def test_remove_hooks_clears_all(self):
        """remove_hooks should clear all forward hooks from all submodules."""
        model = FakeInnerModel()

        # Register hooks on different submodules
        handle1 = model.embed_tokens.register_forward_hook(lambda m, i, o: o)
        handle2 = model.linear.register_forward_hook(lambda m, i, o: o)

        assert len(model.embed_tokens._forward_hooks) == 1
        assert len(model.linear._forward_hooks) == 1

        remove_hooks(model)

        assert len(model.embed_tokens._forward_hooks) == 0
        assert len(model.linear._forward_hooks) == 0

    def test_remove_hooks_preserves_dict_identity(self):
        """remove_hooks should pop entries, not replace the dict object."""
        model = FakeInnerModel()
        original_dict = model.embed_tokens._forward_hooks

        model.embed_tokens.register_forward_hook(lambda m, i, o: o)
        remove_hooks(model)

        # The dict object should be the same (not replaced with a new OrderedDict)
        assert model.embed_tokens._forward_hooks is original_dict

    def test_remove_hooks_noop_when_no_hooks(self):
        """remove_hooks is safe to call when there are no hooks."""
        model = FakeInnerModel()
        remove_hooks(model)  # Should not raise
        assert len(model.embed_tokens._forward_hooks) == 0

    def test_remove_hooks_only_clears_forward_hooks(self):
        """remove_hooks should not affect backward hooks."""
        model = FakeInnerModel()
        bw_handle = model.embed_tokens.register_full_backward_hook(lambda m, gi, go: None)
        fw_handle = model.embed_tokens.register_forward_hook(lambda m, i, o: o)

        remove_hooks(model)

        assert len(model.embed_tokens._forward_hooks) == 0
        assert len(model.embed_tokens._backward_hooks) == 1
        bw_handle.remove()


class TestHandleBasedRemoval:
    """Tests for RegeneratingModel's handle-tracking approach."""

    def test_apply_embeddings_extraction_hook_returns_handle(self):
        """apply_embeddings_extraction_hook should return a valid RemovableHandle."""
        module = SimpleEmbedding()
        data = []
        handle = apply_embeddings_extraction_hook(module, data)
        assert len(module._forward_hooks) == 1
        handle.remove()
        assert len(module._forward_hooks) == 0

    def test_multiple_handles_independent_removal(self):
        """Multiple hooks can be removed independently via their handles."""
        module = SimpleEmbedding()
        h1 = apply_embeddings_extraction_hook(module, [])
        h2 = apply_embeddings_extraction_hook(module, [])
        assert len(module._forward_hooks) == 2

        h1.remove()
        assert len(module._forward_hooks) == 1

        h2.remove()
        assert len(module._forward_hooks) == 0

    def test_double_remove_is_safe(self):
        """Removing an already-removed handle should not raise."""
        module = SimpleEmbedding()
        handle = apply_embeddings_extraction_hook(module, [])
        handle.remove()
        handle.remove()  # Should not raise


class TestRegeneratingModelHookTracking:
    """Integration tests for RegeneratingModel._hook_handles lifecycle."""

    def _make_regenerating_model(self):
        """Create a minimal RegeneratingModel with a real embedding module."""
        from redact.regen.model import RegeneratingModel

        class FakeHFModel:
            def __init__(self):
                self.model_name = "test"
                self.model = FakeInnerModel()
                self.tokenizer = _FakeTokenizer()
                self.model_config = {"embedding_module": "embed_tokens"}
                self.device = torch.device("cpu")
                self.generation_config = None

        hf = FakeHFModel()
        return RegeneratingModel(model=hf)

    def test_hook_handles_start_empty(self):
        model = self._make_regenerating_model()
        assert model._hook_handles == []

    def test_apply_hook_tracks_handle(self):
        model = self._make_regenerating_model()
        model.apply_embeddings_extraction()
        assert len(model._hook_handles) == 1
        assert len(model.model._forward_hooks) == 1

    def test_remove_hooks_clears_tracked_handles(self):
        model = self._make_regenerating_model()
        model.apply_embeddings_extraction()
        assert len(model.model._forward_hooks) == 1

        model.remove_hooks()
        assert len(model._hook_handles) == 0
        assert len(model.model._forward_hooks) == 0

    def test_remove_hooks_then_reapply(self):
        """Simulate the per-batch pattern: remove old hooks, apply new ones."""
        model = self._make_regenerating_model()

        # First batch
        model.apply_embeddings_extraction()
        assert len(model._hook_handles) == 1

        # Second batch — remove old, apply new
        model.remove_hooks()
        model.apply_embeddings_extraction()
        assert len(model._hook_handles) == 1
        assert len(model.model._forward_hooks) == 1

    def test_embeddings_extraction_tracked(self):
        model = self._make_regenerating_model()
        model.apply_embeddings_extraction()
        assert len(model._hook_handles) == 1

        model.remove_hooks()
        assert len(model._hook_handles) == 0


class _FakeEncoding(dict):
    """Mimics BatchEncoding: supports both dict['key'] and dict.key access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class _FakeTokenizer:
    """Minimal tokenizer stub for RegeneratingModel tests."""

    padding_side = "right"

    def __call__(self, text, **kwargs):
        # Return fake token IDs — just split on spaces
        if isinstance(text, str):
            ids = list(range(len(text.split()) + 1))  # +1 for BOS
            return _FakeEncoding({"input_ids": ids})
        return [self(t, **kwargs) for t in text]


# ---------------------------------------------------------------------------
# Issue #3 (continued): _remove_hooks dispatcher in inference.py
# ---------------------------------------------------------------------------


class TestRemoveHooksDispatcher:
    """Tests for the _remove_hooks() function in redact/inference.py."""

    def test_dispatches_to_attacked_model_method(self):
        """_remove_hooks should unwrap ReasoningModel and call RegeneratingModel.remove_hooks()."""
        from redact.inference import _remove_hooks
        from redact.regen.model import RegeneratingModel
        from redact.model.reasoning import ReasoningModel

        class FakeHFModel:
            def __init__(self):
                self.model_name = "test"
                self.model = FakeInnerModel()
                self.tokenizer = _FakeTokenizer()
                self.model_config = {"embedding_module": "embed_tokens"}
                self.device = torch.device("cpu")
                self.generation_config = None

        hf = FakeHFModel()
        attacked = RegeneratingModel(model=hf)

        # Add a hook (via the kept embeddings-extraction utility).
        attacked.apply_embeddings_extraction()
        assert len(attacked._hook_handles) == 1

        # Wrap in ReasoningModel — _remove_hooks should unwrap and use handle-based method
        reasoning = ReasoningModel(attacked)
        _remove_hooks(reasoning)
        assert len(attacked._hook_handles) == 0
        assert len(attacked.model._forward_hooks) == 0

    def test_dispatches_to_standalone_for_plain_model(self):
        """_remove_hooks should use standalone remove_hooks for non-RegeneratingModel."""
        from redact.inference import _remove_hooks

        class FakePlainModel:
            def __init__(self):
                self.model = FakeInnerModel()

        plain = FakePlainModel()
        # Manually add a hook to the inner model
        plain.model.embed_tokens.register_forward_hook(lambda m, i, o: o)
        assert len(plain.model.embed_tokens._forward_hooks) == 1

        _remove_hooks(plain)
        assert len(plain.model.embed_tokens._forward_hooks) == 0
