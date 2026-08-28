from .cosine_similarity import CosineSimilarity
from .entailment_score import EntailmentScore
from .llm_judge_rtofu_score import LLMJudgeRtofuScore
from .rouge_recall import ROUGERecall
from .token_entropy import TokenEntropy

evaluator_classes = {
    c.name: c
    for c in [
        CosineSimilarity,
        EntailmentScore,
        LLMJudgeRtofuScore,
        ROUGERecall,
        TokenEntropy,
    ]
}
