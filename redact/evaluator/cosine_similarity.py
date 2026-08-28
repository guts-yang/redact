import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class CosineSimilarity:
    name = "cosine_similarity"

    def __init__(self, model_name="paraphrase-MiniLM-L6-v2", batch_size=64):
        self.model = SentenceTransformer(
            model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size

    def evaluate(self, answers, generated_answers):
        if not answers:
            return []
        with torch.no_grad():
            gold_embs = self.model.encode(
                list(answers),
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            gen_embs = self.model.encode(
                list(generated_answers),
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        # Pairwise dot product after normalize = cosine similarity.
        sims = np.einsum("ij,ij->i", gold_embs, gen_embs)
        return [max(0.0, float(s)) for s in sims]
