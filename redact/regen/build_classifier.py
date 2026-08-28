"""Factory for constructing LeakClassifier instances from CLI args."""

from __future__ import annotations

from redact.regen.classifier import LeakClassifier


def build_leak_classifier(
    classifier_type: str,
    classifier_path: str,
    classifier_threshold: float | None = None,
    batch_size: int = 32,
) -> LeakClassifier:
    """Construct a :class:`LeakClassifier` from CLI-style arguments.

    Parameters
    ----------
    classifier_type : {"contextual_binary"}
        The only supported value; names the canonical leak detector.
    classifier_path : str
        Path to the trained classifier checkpoint.
    classifier_threshold : float | None
        Binary confidence threshold for contextual_binary.  When None, the
        classifier loads its threshold from
        ``BinaryHeadConfig.binary_threshold`` on the saved checkpoint.
    batch_size : int
        Pipeline / inference batch size.
    """
    if classifier_type == "contextual_binary":
        from redact.regen.classifier import ContextualBinaryLeakClassifier

        return ContextualBinaryLeakClassifier(
            model_path=classifier_path,
            threshold=classifier_threshold,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unsupported classifier_type: {classifier_type!r}. The only "
            f"supported type is 'contextual_binary'. Older checkpoints "
            f"trained as 'twostage' or 'multiclass' detectors are not "
            f"loadable by this release."
        )
