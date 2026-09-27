import argparse
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ConfigSingleton import ConfigSingleton
from logger import Logger_Singleton
from utils import get_model_weight_path, load_model, load_train_valid_dataset, set_seed


SUPPORTED_MODELS = {
    "vgg16",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "inception_v3",
    "resnet50",
}

IMAGE_SIZE_BY_MODEL = {
    "inception_v3": 299,
    "vgg16": 224,
    "mobilenet_v3_small": 224,
    "mobilenet_v3_large": 224,
    "resnet50": 224,
}


@dataclass
class BiasDetectionResult:
    """Results from one bias detection method."""

    method: str
    subgroups: Optional[List[np.ndarray]] = None
    group_losses: Optional[Dict[int, float]] = None
    bias_scores: Optional[np.ndarray] = None
    spurious_correlations: Optional[List[Tuple[str, str]]] = None
    details: Optional[Dict[str, Any]] = None


class BiasDetectionMethod(ABC):
    """Base class for bias detection and correction methods."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    @abstractmethod
    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Detect bias in dataset."""

    @abstractmethod
    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Apply bias correction."""


def _extract_logits(model_output: Any) -> torch.Tensor:
    """Handle models that return tuple-like outputs (for example Inception)."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (list, tuple)) and len(model_output) > 0:
        return model_output[0]
    raise TypeError("Unsupported model output format while extracting logits")


def _embedding_from_forward_hook(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract penultimate features using avgpool when available.
    Falls back to logits if no suitable feature tap exists.
    """
    cache: Dict[str, torch.Tensor] = {}

    hook = None
    if hasattr(model, "avgpool"):
        def _hook(_module: nn.Module, _inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
            cache["feat"] = out

        hook = model.avgpool.register_forward_hook(_hook)

    output = model(inputs)
    logits = _extract_logits(output)

    if hook is not None:
        hook.remove()

    feat = cache.get("feat")
    if feat is None:
        feat = logits

    if feat.ndim > 2:
        feat = torch.flatten(feat, 1)
    return feat


def _safe_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _sanitize_sheet_name(name: str, used: set) -> str:
    invalid_chars = set('[]:*?/\\')
    cleaned = "".join("_" if c in invalid_chars else c for c in str(name))
    cleaned = cleaned[:31] if cleaned else "Sheet"
    base = cleaned
    suffix = 1
    while cleaned.lower() in used:
        tail = str(suffix)
        cleaned = (base[: 31 - len(tail)] + tail)[:31]
        suffix += 1
    used.add(cleaned.lower())
    return cleaned


def _style_worksheet(ws: Any) -> None:
    """Apply a compact style that matches existing reporting sheets."""
    try:
        from openpyxl.styles import Alignment, Font, PatternFill
    except Exception:
        return

    if ws.max_row < 1:
        return

    fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    font = Font(bold=True, color="FFFFFF")

    for cell in ws[1]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(value))
        ws.column_dimensions[col_letter].width = min(max_len + 2, 50)


def _infer_sample_paths_and_labels(loader: DataLoader) -> Tuple[List[str], List[int]]:
    dataset = loader.dataset
    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        paths = [str(x[0]) for x in samples]
        labels = [int(x[1]) for x in samples]
        return paths, labels

    if hasattr(dataset, "getfilelist"):
        paths, labels = dataset.getfilelist()
        return [str(p) for p in paths], [int(y) for y in labels]

    raise ValueError("Unable to infer sample paths from dataset. Expected .samples or .getfilelist()")


def _infer_group_labels_from_parent_folder(paths: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
    parent_names = [os.path.basename(os.path.dirname(p)) for p in paths]
    uniq = sorted(set(parent_names))
    group_to_id = {name: idx for idx, name in enumerate(uniq)}
    labels = np.array([group_to_id[name] for name in parent_names], dtype=np.int64)
    id_to_group = {v: k for k, v in group_to_id.items()}
    return labels, id_to_group


def _resolve_model_path(model_name: str, model_path_arg: str) -> str:
    if os.path.isfile(model_path_arg):
        return model_path_arg
    if os.path.isdir(model_path_arg):
        return get_model_weight_path(model_name, model_path_arg)
    raise FileNotFoundError(f"model_path not found: {model_path_arg}")


class GroupDRO(BiasDetectionMethod):
    """Group Distributionally Robust Optimization (Sagawa et al., ICLR 2020)."""

    def __init__(self, model: nn.Module, group_labels: np.ndarray, device: str = "cpu"):
        super().__init__(model, device)
        self.group_labels = np.asarray(group_labels).astype(np.int64)
        self.num_groups = int(len(np.unique(group_labels)))

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Compute worst-group loss from sample-aligned group labels."""
        group_losses: Dict[int, List[float]] = {i: [] for i in range(self.num_groups)}
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        cursor = 0

        self.model.eval()
        with torch.no_grad():
            for X, y in train_loader:
                logits = _extract_logits(self.model(X.to(self.device)))
                losses = _safe_numpy(loss_fn(logits, y.to(self.device)))

                batch_size = len(losses)
                batch_groups = self.group_labels[cursor : cursor + batch_size]
                cursor += batch_size

                if len(batch_groups) == 0:
                    continue

                for group_id in np.unique(batch_groups):
                    idx = np.where(batch_groups == group_id)[0]
                    if len(idx) > 0:
                        group_losses[int(group_id)].extend(losses[idx].tolist())

        avg_group_losses = {k: float(np.mean(v)) if v else 0.0 for k, v in group_losses.items()}
        worst_group_loss = max(avg_group_losses.values()) if avg_group_losses else 0.0

        return BiasDetectionResult(
            method="GroupDRO",
            group_losses=avg_group_losses,
            bias_scores=np.array([worst_group_loss], dtype=np.float64),
            details={"worst_group_loss": worst_group_loss},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Reweight samples by group (placeholder for now)."""
        return train_loader


class AdversarialDebiasing(BiasDetectionMethod):
    """Remove sensitive attributes from embeddings (Zhang et al., AAAI 2018)."""

    def __init__(self, model: nn.Module, sensitive_attr_dim: int, device: str = "cpu"):
        super().__init__(model, device)
        self.sensitive_attr_dim = int(max(2, sensitive_attr_dim))
        self.adversary: Optional[nn.Module] = None

    def _init_adversary(self, hidden_dim: int) -> None:
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.sensitive_attr_dim),
        ).to(self.device)

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Measure predictability of inferred sensitive groups from embeddings."""
        adv_loss = 0.0
        count = 0
        loss_fn = nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            for X, y in train_loader:
                emb = _embedding_from_forward_hook(self.model, X.to(self.device))
                if self.adversary is None:
                    self._init_adversary(emb.shape[1])
                assert self.adversary is not None
                self.adversary.eval()
                pred = self.adversary(emb)
                # Uses class labels as the available proxy target.
                target = y.to(self.device)
                if pred.shape[1] != self.sensitive_attr_dim:
                    continue
                adv_loss += float(loss_fn(pred, target).item())
                count += 1

        score = adv_loss / count if count > 0 else 0.0
        return BiasDetectionResult(
            method="AdversarialDebiasing",
            bias_scores=np.array([score], dtype=np.float64),
            details={"avg_adv_loss": score},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Train with adversarial objective (placeholder)."""
        return train_loader


class FairBatch(BiasDetectionMethod):
    """Adaptive mini-batch sampling (Roh et al., ICLR 2021)."""

    def __init__(self, model: nn.Module, group_labels: np.ndarray, device: str = "cpu"):
        super().__init__(model, device)
        self.group_labels = np.asarray(group_labels).astype(np.int64)
        self.num_groups = int(len(np.unique(group_labels)))

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Measure group representation imbalance."""
        group_counts = np.bincount(self.group_labels, minlength=self.num_groups)
        ratio = float(group_counts.max() / group_counts.min()) if group_counts.min() > 0 else 0.0

        return BiasDetectionResult(
            method="FairBatch",
            bias_scores=np.array([ratio], dtype=np.float64),
            details={"group_counts": group_counts.tolist(), "imbalance_ratio": ratio},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Create balanced mini-batches (placeholder)."""
        return train_loader


class JTT(BiasDetectionMethod):
    """Just Train Twice (Liu et al., ICML 2021)."""

    def __init__(self, model: nn.Module, device: str = "cpu", overparameterization_factor: float = 1.0):
        super().__init__(model, device)
        self.overparameterization_factor = overparameterization_factor

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Identify high-loss samples."""
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        high_loss_samples: List[float] = []

        with torch.no_grad():
            for X, y in train_loader:
                logits = _extract_logits(self.model(X.to(self.device)))
                losses = loss_fn(logits, y.to(self.device))
                high_loss_samples.extend(_safe_numpy(losses).tolist())

        if high_loss_samples:
            threshold = float(np.percentile(high_loss_samples, 75))
        else:
            threshold = 0.0

        return BiasDetectionResult(
            method="JTT",
            bias_scores=np.array(high_loss_samples, dtype=np.float64),
            details={"high_loss_threshold_p75": threshold},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Retrain focusing on high-loss samples (placeholder)."""
        return train_loader


class LfF(BiasDetectionMethod):
    """Learning from Failure (Nam et al., NeurIPS 2020)."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        super().__init__(model, device)

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Identify bias-conflicting samples using median loss split."""
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        losses: List[float] = []

        with torch.no_grad():
            for X, y in train_loader:
                logits = _extract_logits(self.model(X.to(self.device)))
                batch_losses = loss_fn(logits, y.to(self.device))
                losses.extend(_safe_numpy(batch_losses).tolist())

        threshold = float(np.percentile(losses, 50)) if losses else 0.0
        return BiasDetectionResult(
            method="LfF",
            bias_scores=np.array(losses, dtype=np.float64),
            details={"conflict_threshold_p50": threshold},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Train on bias-conflicting samples (placeholder)."""
        return train_loader


class EIIL(BiasDetectionMethod):
    """Estimate Invariant Independences with Latent confounder (Creager et al., ICML 2021)."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        super().__init__(model, device)

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Infer hidden subgroups from embeddings with k-means."""
        embeddings = []

        self.model.eval()
        with torch.no_grad():
            for X, _ in train_loader:
                emb = _embedding_from_forward_hook(self.model, X.to(self.device))
                embeddings.append(_safe_numpy(emb))

        if not embeddings:
            return BiasDetectionResult(method="EIIL", subgroups=[])

        matrix = np.vstack(embeddings)
        from sklearn.cluster import KMeans

        n_clusters = min(4, max(2, matrix.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(matrix)

        subgroups = [np.where(labels == i)[0] for i in range(n_clusters)]

        return BiasDetectionResult(
            method="EIIL",
            subgroups=subgroups,
            details={"n_clusters": n_clusters},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Train invariant predictor (placeholder)."""
        return train_loader


class SpectralDecoupling(BiasDetectionMethod):
    """Prevent shortcut learning (Pezeshki et al., ICLR 2021)."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        super().__init__(model, device)

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Measure feature spectrum condition number."""
        self.model.eval()
        features = []

        with torch.no_grad():
            for X, _ in train_loader:
                feat = _embedding_from_forward_hook(self.model, X.to(self.device))
                features.append(_safe_numpy(feat))

        if not features:
            cond = float("inf")
        else:
            matrix = np.vstack(features)
            _u, s, _vt = np.linalg.svd(matrix, full_matrices=False)
            cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")

        return BiasDetectionResult(
            method="SpectralDecoupling",
            bias_scores=np.array([cond], dtype=np.float64),
            details={"condition_number": cond},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Regularize feature spectrum (placeholder)."""
        return train_loader


class DIM(BiasDetectionMethod):
    """Discover and Mitigate multiple biased subgroups (Zhang et al., CVPR 2024)."""

    def __init__(self, model: nn.Module, device: str = "cpu", num_subgroups: int = 4):
        super().__init__(model, device)
        self.num_subgroups = max(2, int(num_subgroups))

    def detect(self, train_loader: DataLoader) -> BiasDetectionResult:
        """Discover hidden biased subgroups with k-means clustering."""
        embeddings = []

        self.model.eval()
        with torch.no_grad():
            for X, _ in train_loader:
                emb = _embedding_from_forward_hook(self.model, X.to(self.device))
                embeddings.append(_safe_numpy(emb))

        if not embeddings:
            return BiasDetectionResult(method="DIM", subgroups=[])

        matrix = np.vstack(embeddings)
        from sklearn.cluster import KMeans

        k = min(self.num_subgroups, max(2, matrix.shape[0]))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        subgroup_labels = kmeans.fit_predict(matrix)
        subgroups = [np.where(subgroup_labels == i)[0] for i in range(k)]

        return BiasDetectionResult(
            method="DIM",
            subgroups=subgroups,
            details={"n_clusters": k},
        )

    def correct(self, train_loader: DataLoader) -> DataLoader:
        """Mitigate discovered subgroup biases (placeholder)."""
        return train_loader


class BiasDetectionFramework:
    """Main framework for dataset bias detection and correction."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.methods: Dict[str, BiasDetectionMethod] = {}
        self.results: Dict[str, BiasDetectionResult] = {}

    def register_method(self, name: str, method: BiasDetectionMethod) -> None:
        self.methods[name] = method

    def detect_all(self, train_loader: DataLoader) -> Dict[str, BiasDetectionResult]:
        self.results = {}
        for name, method in self.methods.items():
            self.results[name] = method.detect(train_loader)
        return self.results

    def correct_all(self, train_loader: DataLoader) -> Dict[str, DataLoader]:
        corrected_loaders = {}
        for name, method in self.methods.items():
            corrected_loaders[name] = method.correct(train_loader)
        return corrected_loaders

    def report(self) -> str:
        report = "=== Bias Detection Report ===\n"
        for name, result in self.results.items():
            report += f"\n{name} ({result.method}):\n"
            if result.bias_scores is not None:
                report += f"  Bias Score: {result.bias_scores}\n"
            if result.group_losses is not None:
                report += f"  Group Losses: {result.group_losses}\n"
            if result.subgroups is not None:
                report += f"  Subgroups: {len(result.subgroups)} detected\n"
        return report


def _build_summary_rows(results: Dict[str, BiasDetectionResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, result in results.items():
        scores = result.bias_scores if result.bias_scores is not None else np.array([], dtype=np.float64)
        row = {
            "registry_name": key,
            "method": result.method,
            "num_scores": int(len(scores)),
            "score_mean": float(np.mean(scores)) if len(scores) else np.nan,
            "score_max": float(np.max(scores)) if len(scores) else np.nan,
            "score_min": float(np.min(scores)) if len(scores) else np.nan,
            "num_group_losses": int(len(result.group_losses or {})),
            "num_subgroups": int(len(result.subgroups or [])),
        }
        if result.details:
            for d_key, d_val in result.details.items():
                if isinstance(d_val, (int, float, str, bool)):
                    row[d_key] = d_val
        rows.append(row)
    return rows


def write_results_to_excel(
    output_path: str,
    run_params: Dict[str, Any],
    group_mapping: Dict[int, str],
    sample_df: pd.DataFrame,
    results: Dict[str, BiasDetectionResult],
    text_report: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    used_names: set = set()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        run_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in run_params.items()])
        run_sheet = _sanitize_sheet_name("Run_Parameters", used_names)
        run_df.to_excel(writer, sheet_name=run_sheet, index=False)

        map_df = pd.DataFrame(
            [{"group_id": gid, "group_name": gname} for gid, gname in sorted(group_mapping.items())]
        )
        map_sheet = _sanitize_sheet_name("Group_Mapping", used_names)
        map_df.to_excel(writer, sheet_name=map_sheet, index=False)

        sample_sheet = _sanitize_sheet_name("Sample_Diagnostics", used_names)
        sample_df.to_excel(writer, sheet_name=sample_sheet, index=False)

        summary_df = pd.DataFrame(_build_summary_rows(results))
        summary_sheet = _sanitize_sheet_name("Method_Summary", used_names)
        summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)

        text_sheet = _sanitize_sheet_name("Report_Text", used_names)
        pd.DataFrame({"report": text_report.splitlines()}).to_excel(writer, sheet_name=text_sheet, index=False)

        for key, result in results.items():
            base = f"{key}_details"
            rows: List[Dict[str, Any]] = []

            if result.group_losses is not None:
                for gid, loss in sorted(result.group_losses.items()):
                    rows.append({"detail_type": "group_loss", "group_id": int(gid), "value": float(loss)})

            if result.bias_scores is not None:
                for idx, score in enumerate(result.bias_scores.tolist()):
                    rows.append({"detail_type": "bias_score", "index": idx, "value": float(score)})

            if result.subgroups is not None:
                for idx, subgroup in enumerate(result.subgroups):
                    rows.append(
                        {
                            "detail_type": "subgroup",
                            "subgroup_id": idx,
                            "subgroup_size": int(len(subgroup)),
                            "indices": ",".join(str(int(x)) for x in subgroup[:300]),
                        }
                    )

            if result.details:
                for d_key, d_val in result.details.items():
                    rows.append({"detail_type": "meta", "key": d_key, "value": str(d_val)})

            details_df = pd.DataFrame(rows)
            sheet_name = _sanitize_sheet_name(base, used_names)
            details_df.to_excel(writer, sheet_name=sheet_name, index=False)

        for ws in writer.book.worksheets:
            _style_worksheet(ws)


def _parse_methods(methods_arg: str) -> List[str]:
    if methods_arg.strip().lower() == "all":
        return [
            "groupdro",
            "fairbatch",
            "adversarialdebiasing",
            "jtt",
            "lff",
            "eiil",
            "spectraldecoupling",
            "dim",
        ]
    return [m.strip().lower() for m in methods_arg.split(",") if m.strip()]


def _register_methods(
    framework: BiasDetectionFramework,
    method_names: List[str],
    group_labels: np.ndarray,
    num_classes: int,
    num_subgroups: int,
    device: str,
) -> None:
    registry: Dict[str, BiasDetectionMethod] = {
        "groupdro": GroupDRO(framework.model, group_labels=group_labels, device=device),
        "fairbatch": FairBatch(framework.model, group_labels=group_labels, device=device),
        "adversarialdebiasing": AdversarialDebiasing(
            framework.model,
            sensitive_attr_dim=num_classes,
            device=device,
        ),
        "jtt": JTT(framework.model, device=device),
        "lff": LfF(framework.model, device=device),
        "eiil": EIIL(framework.model, device=device),
        "spectraldecoupling": SpectralDecoupling(framework.model, device=device),
        "dim": DIM(framework.model, device=device, num_subgroups=num_subgroups),
    }

    for name in method_names:
        if name not in registry:
            raise ValueError(f"Unknown method '{name}'.")
        framework.register_method(name, registry[name])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bias detection runner with Excel export")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--store_results", type=str, default="./results")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_subgroups", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--excel_name", type=str, default=None)
    return parser


def run_bias_detection(args: argparse.Namespace) -> str:
    model_name = args.model_name.strip()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose one of: {sorted(SUPPORTED_MODELS)}"
        )

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file does not exist: {args.config_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = os.path.join(args.store_results, model_name)
    os.makedirs(result_root, exist_ok=True)

    log_path = os.path.join(result_root, f"bias_detection_{timestamp}.log")
    logger = Logger_Singleton(log_path)

    config = ConfigSingleton(args.config_file)
    seed = int(getattr(config, "SEED", 132))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    batch_size = int(args.batch_size) if args.batch_size else int(config.BATCH_SIZE)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    model_path = _resolve_model_path(model_name, args.model_path)
    logger.info(f"Using model_path={model_path}")
    model = load_model(model_name, model_path).to(device)
    model.eval()

    train_loader, _valid_loader, _train_tf, _valid_tf, class_names = load_train_valid_dataset(
        model_name,
        config.CLASSIFICATION_DATA_BASE_PATH,
        batch_size,
        random_state=seed,
    )

    ordered_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    sample_paths, sample_labels = _infer_sample_paths_and_labels(ordered_loader)
    group_labels, id_to_group = _infer_group_labels_from_parent_folder(sample_paths)

    sample_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(sample_paths)),
            "file_path": sample_paths,
            "class_label": sample_labels,
            "class_name": [class_names[y] if y < len(class_names) else str(y) for y in sample_labels],
            "group_id": group_labels,
            "group_name": [id_to_group[int(g)] for g in group_labels],
        }
    )

    framework = BiasDetectionFramework(model=model, device=device)
    method_names = _parse_methods(args.methods)
    _register_methods(
        framework,
        method_names,
        group_labels=group_labels,
        num_classes=max(2, len(class_names)),
        num_subgroups=args.num_subgroups,
        device=device,
    )

    logger.info(f"Running methods: {method_names}")
    results = framework.detect_all(ordered_loader)
    text_report = framework.report()

    excel_name = args.excel_name if args.excel_name else f"bias_detection_report_{model_name}_{timestamp}.xlsx"
    output_path = os.path.join(result_root, excel_name)

    run_params: Dict[str, Any] = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_path": model_path,
        "config_file": args.config_file,
        "data_base_path": config.CLASSIFICATION_DATA_BASE_PATH,
        "target_classes": list(config.TARGET_CLASS_LIST),
        "batch_size": batch_size,
        "device": device,
        "methods": ",".join(method_names),
        "seed": seed,
        "num_samples": len(sample_paths),
        "group_inference_rule": "parent_folder_name",
        "image_size": IMAGE_SIZE_BY_MODEL.get(model_name, 224),
    }

    write_results_to_excel(
        output_path=output_path,
        run_params=run_params,
        group_mapping=id_to_group,
        sample_df=sample_df,
        results=results,
        text_report=text_report,
    )

    logger.info(f"Bias detection report written to {output_path}")
    print(text_report)
    print(f"Report path: {output_path}")
    return output_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_bias_detection(args)


if __name__ == "__main__":
    main()
