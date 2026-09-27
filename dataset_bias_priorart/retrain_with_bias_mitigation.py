import argparse
import copy
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ConfigSingleton import ConfigSingleton
from logger import Logger_Singleton
from utils import get_model_weight_path, load_model, load_train_valid_dataset, set_seed

try:
    from dataset_bias_priorart.dataset_bias_detection_methods import (
        AdversarialDebiasing,
        DIM,
        EIIL,
        FairBatch,
        GroupDRO,
        JTT,
        LfF,
        SpectralDecoupling,
        SUPPORTED_MODELS,
    )
except Exception:
    from dataset_bias_detection_methods import (
        AdversarialDebiasing,
        DIM,
        EIIL,
        FairBatch,
        GroupDRO,
        JTT,
        LfF,
        SpectralDecoupling,
        SUPPORTED_MODELS,
    )

from dataset_bias_priorart.retrain_helpers import (
    infer_group_labels_from_parent_folder,
    infer_sample_paths_and_labels,
    make_loader,
    train_one_method,
    write_retraining_excel,
)


def parse_methods(methods_arg: str) -> List[str]:
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


def resolve_model_path(model_name: str, model_path_arg: str) -> str:
    if os.path.isfile(model_path_arg):
        return model_path_arg
    if os.path.isdir(model_path_arg):
        return get_model_weight_path(model_name, model_path_arg)
    raise FileNotFoundError(f"model_path not found: {model_path_arg}")


def build_method(
    method_name: str,
    model: torch.nn.Module,
    group_labels: np.ndarray,
    num_classes: int,
    num_subgroups: int,
    device: str,
):
    if method_name == "groupdro":
        return GroupDRO(model, group_labels=group_labels, device=device)
    if method_name == "fairbatch":
        return FairBatch(model, group_labels=group_labels, device=device)
    if method_name == "adversarialdebiasing":
        return AdversarialDebiasing(model, sensitive_attr_dim=max(2, num_classes), device=device)
    if method_name == "jtt":
        return JTT(model, device=device)
    if method_name == "lff":
        return LfF(model, device=device)
    if method_name == "eiil":
        return EIIL(model, device=device)
    if method_name == "spectraldecoupling":
        return SpectralDecoupling(model, device=device)
    if method_name == "dim":
        return DIM(model, device=device, num_subgroups=num_subgroups)
    raise ValueError(f"Unknown method: {method_name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain model with all bias mitigation methods")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--store_results", type=str, default="./results")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_subgroups", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--excel_name", type=str, default=None)
    parser.add_argument("--save_state_dict", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Tuple[str, List[str], List[str]]:
    model_name = args.model_name.strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {sorted(SUPPORTED_MODELS)}")

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file does not exist: {args.config_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = os.path.join(args.store_results, model_name)
    os.makedirs(result_root, exist_ok=True)
    model_out_dir = os.path.join(result_root, "retrained_models")
    os.makedirs(model_out_dir, exist_ok=True)

    log_path = os.path.join(result_root, f"bias_mitigation_retrain_{timestamp}.log")
    logger = Logger_Singleton(log_path)

    config = ConfigSingleton(args.config_file)
    seed = int(getattr(config, "SEED", 132))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(args.batch_size) if args.batch_size else int(config.BATCH_SIZE)
    epochs = int(args.epochs) if args.epochs else int(config.EPOCHS)
    learning_rate = float(args.learning_rate) if args.learning_rate else float(config.LEARNING_RATE)

    model_path = resolve_model_path(model_name, args.model_path)
    base_model = load_model(model_name, model_path).to(device)
    base_model.eval()

    train_loader, valid_loader, _tt, _vt, class_names = load_train_valid_dataset(
        model_name,
        config.CLASSIFICATION_DATA_BASE_PATH,
        batch_size,
        random_state=seed,
    )

    sample_paths, _sample_labels = infer_sample_paths_and_labels(train_loader.dataset)
    if len(sample_paths) == 0:
        raise ValueError("No training samples found.")

    group_labels, id_to_group = infer_group_labels_from_parent_folder(sample_paths)

    ordered_loader = make_loader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    method_names = parse_methods(args.methods)
    detection_results: Dict[str, Any] = {}
    training_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    succeeded: List[str] = []
    failed: List[str] = []

    for idx, method_name in enumerate(method_names):
        logger.info(f"Starting method={method_name}")
        start_ts = time.time()
        try:
            work_model = copy.deepcopy(base_model).to(device)
            method = build_method(
                method_name,
                work_model,
                group_labels=group_labels,
                num_classes=max(2, len(class_names)),
                num_subgroups=args.num_subgroups,
                device=device,
            )

            detect_result = method.detect(ordered_loader)
            detection_results[method_name] = detect_result

            method_seed = seed + (idx * 97)
            set_seed(method_seed)
            random.seed(method_seed)
            np.random.seed(method_seed)

            method_train_loader = make_loader(
                train_loader.dataset,
                batch_size=batch_size,
                shuffle=True,
                seed=method_seed,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
            )
            method_valid_loader = make_loader(
                valid_loader.dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=method_seed,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
            )

            corrected_loader = method.correct(method_train_loader)
            train_metrics = train_one_method(
                work_model,
                corrected_loader,
                method_valid_loader,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                max_grad_norm=7.0,
            )

            model_file = os.path.join(model_out_dir, f"retrained_{model_name}_{method_name}_{timestamp}.pth")
            if args.save_state_dict:
                torch.save(work_model.state_dict(), model_file)
            else:
                torch.save(work_model, model_file)

            runtime_sec = time.time() - start_ts
            training_rows.append(
                {
                    "method": method_name,
                    "train_loss": float(train_metrics["train_loss"]),
                    "valid_loss": float(train_metrics["valid_loss"]),
                    "valid_accuracy": float(train_metrics["valid_accuracy"]),
                    "runtime_sec": float(runtime_sec),
                    "model_file": model_file,
                }
            )
            succeeded.append(method_name)
            logger.info(f"Completed method={method_name}, accuracy={train_metrics['valid_accuracy']:.4f}")

        except Exception as exc:
            failed.append(method_name)
            runtime_sec = time.time() - start_ts
            logger.error(f"Method failed: {method_name}. Error: {exc}")
            failed_rows.append(
                {
                    "method": method_name,
                    "error": str(exc),
                    "runtime_sec": float(runtime_sec),
                }
            )

    excel_name = args.excel_name if args.excel_name else f"bias_mitigation_retraining_{model_name}_{timestamp}.xlsx"
    excel_path = os.path.join(result_root, excel_name)

    run_params: Dict[str, Any] = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_path": model_path,
        "config_file": args.config_file,
        "data_base_path": config.CLASSIFICATION_DATA_BASE_PATH,
        "target_classes": list(config.TARGET_CLASS_LIST),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": device,
        "methods": ",".join(method_names),
        "seed": seed,
        "num_train_samples": len(sample_paths),
        "group_inference_rule": "parent_folder_name",
        "save_state_dict": bool(args.save_state_dict),
        "succeeded_methods": ",".join(succeeded),
        "failed_methods": ",".join(failed),
    }

    write_retraining_excel(
        output_path=excel_path,
        run_params=run_params,
        group_mapping=id_to_group,
        detection_results=detection_results,
        training_rows=training_rows,
        failed_rows=failed_rows,
    )

    logger.info(f"Retraining summary excel written: {excel_path}")
    logger.info(f"Succeeded methods: {succeeded}")
    logger.info(f"Failed methods: {failed}")
    print(f"Excel summary: {excel_path}")
    print(f"Succeeded methods: {succeeded}")
    print(f"Failed methods: {failed}")
    return excel_path, succeeded, failed


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
