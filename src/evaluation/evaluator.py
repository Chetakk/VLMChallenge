"""
Evaluator Module

Evaluation pipeline for VLM models.
"""

from . import metrics


class ModelEvaluator:
    """Evaluates VLM models on various datasets and metrics."""

    def __init__(self, model, device="cuda"):
        """Initialize evaluator."""
        self.model = model
        self.device = device

    def evaluate(self, test_dataset):
        """Run full evaluation on test dataset."""
        pass

    def evaluate_on_metric(self, test_dataset, metric_fn):
        """Evaluate using specific metric function."""
        pass
