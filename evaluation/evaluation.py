import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# arg_parse.py lives in ../training/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))

# Import rrg_eval modules
import rrg_eval.chexbert
import rrg_eval.rouge
import rrg_eval.f1radgraph
from rrg_eval.f1radgraph import F1RadGraphv2

from arg_parse import parse_evaluation_args

try:
    import evaluate
    from sacrebleu.metrics import BLEU
except ImportError:
    print("Warning: 'evaluate' or 'sacrebleu' not installed. Some metrics may not be available.")
    evaluate = None
    BLEU = None


def bleu4(predictions, references, bootstrap_ci: bool = False):
    """Compute BLEU-4 score."""
    if bootstrap_ci and BLEU is not None:
        ret = BLEU().corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        return {"median": ret.score, "mean": ret._mean, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    elif evaluate is not None:
        return evaluate.load("bleu").compute(predictions=predictions, references=references)["bleu"]
    else:
        raise ImportError("Please install 'evaluate' package: pip install evaluate")


def bleu1(predictions, references, bootstrap_ci: bool = False):
    """Compute BLEU-1 score."""
    if bootstrap_ci and BLEU is not None:
        ret = BLEU(max_ngram_order=1).corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        return {"median": ret.score, "mean": ret._mean, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    elif evaluate is not None:
        return evaluate.load("bleu").compute(predictions=predictions, references=references, max_order=1)["bleu"]
    else:
        raise ImportError("Please install 'evaluate' package: pip install evaluate")


def rougel(predictions, references, bootstrap_ci: bool = False):
    """Compute ROUGE-L score."""
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rougeL"])["rougeL"]
    elif evaluate is not None:
        return evaluate.load("rouge").compute(predictions=predictions, references=references)["rougeL"]
    else:
        raise ImportError("Please install 'evaluate' package: pip install evaluate")


def rouge2(predictions, references, bootstrap_ci: bool = False):
    """Compute ROUGE-2 score."""
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rouge2"])["rouge2"]
    elif evaluate is not None:
        return evaluate.load("rouge").compute(predictions=predictions, references=references)["rouge2"]
    else:
        raise ImportError("Please install 'evaluate' package: pip install evaluate")


def radgraph(predictions, references, bootstrap_ci: bool = False):
    """Compute RadGraph F1 score."""
    if bootstrap_ci:
        reward_list = F1RadGraphv2(reward_level="partial", batch_size=1)(hyps=predictions, refs=references)[1]
        bs = rrg_eval.f1radgraph.bootstrap_confidence_interval(reward_list, n_resamples=500)
        return {
            "median": np.median(bs.bootstrap_distribution),
            "mean": np.mean(bs.bootstrap_distribution), 
            "std": np.std(bs.bootstrap_distribution),
            "ci_l": bs.confidence_interval.low,
            "ci_h": bs.confidence_interval.high,
        }
    else:
        return F1RadGraphv2(reward_level="partial", batch_size=1)(hyps=predictions, refs=references)[0]


def chexbert(predictions, references, bootstrap_ci: bool = False):
    """Compute CheXbert metrics."""
    return rrg_eval.chexbert.evaluate(predictions, references, include_original=False, bootstrap_ci=bootstrap_ci)


SCORER_NAME_TO_CLASS = {
    "ROUGE-L": rougel,
    "ROUGE-2": rouge2,
    "BLEU-4": bleu4,
    "BLEU-1": bleu1,
    "F1-RadGraph": radgraph,
    "CheXbert": chexbert,
}


class ReportGenerationEvaluator:
    """Evaluator for radiology report generation."""

    def __init__(self, scorers=['CheXbert'], bootstrap_ci: bool = False):
        self.bootstrap_ci = bootstrap_ci
        self.scorers = {}

        for scorer_name in scorers:
            if scorer_name in SCORER_NAME_TO_CLASS:
                self.scorers[scorer_name] = SCORER_NAME_TO_CLASS[scorer_name]
            else:
                raise NotImplementedError(f'scorer of type {scorer_name} not implemented')

    def evaluate(self, predictions, references):
        """Run all scorers on predictions and references."""
        assert len(predictions) == len(references), \
            f'Length of predictions {len(predictions)} and references {len(references)} must match.'

        scores = {}

        for scorer_name, scorer in (pbar := tqdm(self.scorers.items())):
            pbar.set_description(f"Computing {scorer_name}")
            scorer_scores = scorer(predictions, references, self.bootstrap_ci)
            scores[scorer_name] = scorer_scores

        self.postprocess_eval(scores)
        return scores

    def postprocess_eval(self, scores):
        """Postprocess evaluation scores to extract specific metrics."""
        if self.bootstrap_ci:
            keys = ("median", "ci_l", "mean", "std", "ci_h")
            for name in list(scores.keys()):
                if name == "CheXbert":
                    metrics = scores.pop(name)
                    scores["Micro-F1-14"] = {k: metrics[0]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14"] = {k: metrics[0]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5"] = {k: metrics[1]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5"] = {k: metrics[1]["macro avg"][k] for k in keys}
                    scores["Micro-F1-14+"] = {k: metrics[2]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14+"] = {k: metrics[2]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5+"] = {k: metrics[3]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5+"] = {k: metrics[3]["macro avg"][k] for k in keys}
                    scores["breakdown-"] = metrics[0]
                    scores["breakdown+"] = metrics[2]
                    scores["chexbert_metrics"] = metrics[-1]
                elif name == "F1-RadGraph":
                    scores["F1-RadGraph"] = scores.pop(name)
        else:
            for name in list(scores.keys()):
                if name == "CheXbert":
                    metrics = scores.pop(name)
                    scores["Micro-F1-14"] = metrics[0]["micro avg"]["f1-score"]
                    scores["Macro-F1-14"] = metrics[0]["macro avg"]["f1-score"]
                    scores["Micro-F1-5"] = metrics[1]["micro avg"]["f1-score"]
                    scores["Macro-F1-5"] = metrics[1]["macro avg"]["f1-score"]
                    scores["Micro-F1-14+"] = metrics[2]["micro avg"]["f1-score"]
                    scores["Macro-F1-14+"] = metrics[2]["macro avg"]["f1-score"]
                    scores["Micro-F1-5+"] = metrics[3]["micro avg"]["f1-score"]
                    scores["Macro-F1-5+"] = metrics[3]["macro avg"]["f1-score"]
                    scores["breakdown-"] = metrics[0]
                    scores["breakdown+"] = metrics[2]
                    scores["chexbert_metrics"] = metrics[-1]
                elif name == "F1-RadGraph":
                    scores["F1-RadGraph"] = scores.pop(name)["f1-radgraph"]


def load_results(filepath: str):
    """Load predictions and references from JSONL or JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    predictions, references = [], []

    # ---- Try JSON first (dict or list) ----
    try:
        obj = json.loads(raw)

        # Case 1: list of records
        if isinstance(obj, list):
            records = obj

        # Case 2: dict with a key holding list of records
        elif isinstance(obj, dict):
            # try common keys
            if "data" in obj and isinstance(obj["data"], list):
                records = obj["data"]
            elif "results" in obj and isinstance(obj["results"], list):
                records = obj["results"]
            elif "predictions" in obj and isinstance(obj["predictions"], list):
                records = obj["predictions"]
            else:
                # single record dict
                records = [obj]
        else:
            raise ValueError("Unsupported JSON top-level type")

        for rec in records:
            if not isinstance(rec, dict):
                continue
            pred = rec.get("text", rec.get("prediction", ""))
            ref  = rec.get("ground_truth", rec.get("reference", ""))
            predictions.append(pred)
            references.append(ref)

        return predictions, references

    except json.JSONDecodeError:
        pass

    # ---- Fallback: JSONL ----
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            predictions.append(data.get("text", data.get("prediction", "")))
            references.append(data.get("ground_truth", data.get("reference", "")))

    return predictions, references


def print_results(results, bootstrap_ci, total_reports):
    """Print evaluation results in a formatted table."""
    print("\n")
    print(f"Total reports: {total_reports}\n")

    print("========== Main Results ==========")
    if bootstrap_ci:
        main_results = pd.DataFrame.from_dict({
            k: v for k, v in results.items() if k not in ("breakdown+", "breakdown-", "chexbert_metrics")
        })
        # Print only available columns
        available_cols = [c for c in [
            "Micro-F1-14", "Micro-F1-5", "Macro-F1-14", "Macro-F1-5",
            "Micro-F1-14+", "Micro-F1-5+", "Macro-F1-14+", "Macro-F1-5+",
            "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"
        ] if c in main_results.columns]
        print(main_results[available_cols])
    else:
        main_results = pd.DataFrame.from_dict(
            {k: v for k, v in results.items() if not isinstance(v, dict)},
            orient='index'
        )
        available_cols = [c for c in [
            "Micro-F1-14", "Micro-F1-5", "Macro-F1-14", "Macro-F1-5",
            "Micro-F1-14+", "Micro-F1-5+", "Macro-F1-14+", "Macro-F1-5+",
            "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"
        ] if c in main_results.index]
        print(main_results.T[available_cols] if available_cols else main_results.T)
    print("")

    return main_results


def save_results(results, output_dir, bootstrap_ci):
    """Save evaluation results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save main results
    if bootstrap_ci:
        main_results = pd.DataFrame.from_dict({
            k: v for k, v in results.items() if k not in ("breakdown+", "breakdown-", "chexbert_metrics")
        })
    else:
        main_results = pd.DataFrame.from_dict(
            {k: v for k, v in results.items() if not isinstance(v, dict)},
            orient='index'
        )

    main_results.to_csv(os.path.join(output_dir, "main_results.csv"))
    print(f"Saved main results to {os.path.join(output_dir, 'main_results.csv')}")

    # Save CheXbert breakdowns if available
    if "breakdown+" in results:
        from rrg_eval.factuality_utils import CONDITIONS

        breakdown_p = pd.DataFrame(results["breakdown+"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ['f1-score', 'precision', 'recall', 'support']
        ]
        breakdown_p.to_csv(os.path.join(output_dir, "chexbert_breakdown_positive.csv"))
        print(f"Saved CheXbert breakdown (uncertain as positive) to {os.path.join(output_dir, 'chexbert_breakdown_positive.csv')}")

        print("\n========== CheXbert F1 (uncertain as positive) ==========")
        print(breakdown_p)
        print("")

    if "breakdown-" in results:
        from rrg_eval.factuality_utils import CONDITIONS

        breakdown_n = pd.DataFrame(results["breakdown-"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ['f1-score', 'precision', 'recall', 'support']
        ]
        breakdown_n.to_csv(os.path.join(output_dir, "chexbert_breakdown_negative.csv"))
        print(f"Saved CheXbert breakdown (uncertain as negative) to {os.path.join(output_dir, 'chexbert_breakdown_negative.csv')}")

        print("========== CheXbert F1 (uncertain as negative) ==========")
        print(breakdown_n)
        print("")

    # Save all results as JSON
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_json = convert_to_serializable(results)
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved all results to {os.path.join(output_dir, 'all_results.json')}")


def main():


    args = parse_evaluation_args()

    print(f"Loading results from {args.results_file}...")
    predictions, references = load_results(args.results_file)
    print(f"Loaded {len(predictions)} predictions and {len(references)} references")

    # Filter out empty predictions or references
    filtered_preds = []
    filtered_refs = []
    for pred, ref in zip(predictions, references):
        if pred.strip() and ref.strip():
            filtered_preds.append(pred)
            filtered_refs.append(ref)

    if len(filtered_preds) < len(predictions):
        print(f"Filtered out {len(predictions) - len(filtered_preds)} empty predictions/references")
        predictions = filtered_preds
        references = filtered_refs

    print("After filtering:", len(predictions))
    print("Example pred:", repr(predictions[0][:200]) if predictions else "NONE")
    print("Example ref :", repr(references[0][:200]) if references else "NONE")
    print(f"\nEvaluating with scorers: {args.scorers}")
    print(f"Bootstrap CI: {args.bootstrap_ci}")

    evaluator = ReportGenerationEvaluator(scorers=args.scorers, bootstrap_ci=args.bootstrap_ci)
    results = evaluator.evaluate(predictions, references)

    # Print and save results
    main_results = print_results(results, args.bootstrap_ci, len(predictions))
    save_results(results, args.output_dir, args.bootstrap_ci)

    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()