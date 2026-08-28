from rouge_score import rouge_scorer


class ROUGERecall:
    """ROUGE recall of each generation against its reference answer.

    The class-level :attr:`name` (``"rouge_recall"``) is the key under
    which this evaluator is registered in
    :mod:`redact.evaluator`. The instance shadows it with a
    mode-qualified name (e.g. ``"rougeL_recall"``) so the per-result
    metric keys identify which ROUGE variant produced them — this is
    the suffix :func:`redact.utils.compute_mu` looks for.
    """

    name = "rouge_recall"

    def __init__(self, mode="rougeL"):
        self.mode = mode
        self.name = f"{mode}_recall"
        self._scorer = rouge_scorer.RougeScorer([mode], use_stemmer=True)

    def evaluate(self, answers, generated_answers):
        """Return per-pair ROUGE recall as a list of floats.

        Scores each ``(reference, generation)`` pair with the configured
        ROUGE variant and collects the recall component, preserving the
        input order.
        """
        return [
            self._scorer.score(reference, generation)[self.mode].recall
            for reference, generation in zip(answers, generated_answers)
        ]
