# Copyright [2025] [Srikanth KS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import argparse
import random
from datetime import datetime
from itertools import combinations

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from logger import Logger_Singleton
from custom_dataloader import MultiClassImageDataset
from ConfigSingleton import ConfigSingleton
from cav_registry import CAVRegistry
from utils import (
    get_class_folder_dicts,
    evaluate_accuracy,
    plot_loss_figure,
    save_statistics,
    compute_avg_confidence,
    get_model_weight_path,
    get_base_model_image_size,
    get_model_layers,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

RANDOM_STATE = 132


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    np.random.seed(RANDOM_STATE + worker_id)
    random.seed(RANDOM_STATE + worker_id)


set_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Activation hook — no global state; stores into a caller-supplied dict
# ---------------------------------------------------------------------------

class ActivationHook:
    """Captures activations for a specific layer into a provided dictionary."""
    def __init__(self, layer_name: str, store: dict):
        self.layer_name = layer_name
        self.store = store
    
    def __call__(self, model, inputs, output):
        self.store[self.layer_name] = output


def get_activation(layer_name: str, store: dict):
    """Returns an ActivationHook callable for capturing layer outputs."""
    return ActivationHook(layer_name, store)


def _extract_images_and_labels(batch):
    if isinstance(batch, (tuple, list)):
        imgs = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        return imgs, labels
    return batch, None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(model, validation_loader, target_idx_list, logging):
    try:
        accuracy, precision, recall, f1, class_results = evaluate_accuracy(
            model, validation_loader
        )
        avg_conf = compute_avg_confidence(model, validation_loader, target_idx_list)
        return accuracy, precision, recall, f1, avg_conf, class_results
    except Exception as exc:
        logging.error(f"Metrics computation failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# CAV loading via registry
# ---------------------------------------------------------------------------

def _load_target_cavs(
    registry: CAVRegistry,
    model_name: str,
    layer_names: list,
    concept_name: str,
    logging,
) -> dict:
    target_cavs = {}
    for layer in layer_names:
        try:
            raw = registry.get_cav(model_name, layer, concept_name)
            if not isinstance(raw, torch.Tensor):
                raw = torch.tensor(raw, dtype=torch.float32)
            cav = F.normalize(
                raw.to(DEVICE, dtype=torch.float32).view(1, -1), p=2, dim=1
            )
            target_cavs[layer] = cav
            logging.info(
                f"  CAV loaded: model={model_name} layer={layer} "
                f"concept={concept_name} shape={cav.shape}"
            )
        except (FileNotFoundError, KeyError) as exc:
            logging.warning(
                f"  CAV not found model={model_name} layer={layer} "
                f"concept={concept_name}: {exc}"
            )
    return target_cavs


# ---------------------------------------------------------------------------
# TCAV scores
# ---------------------------------------------------------------------------

def _compute_tcav_scores(
    model,
    target_cavs: dict,
    loader,
    target_idx: int,
    logging,
) -> dict:
    scores = {}
    activation_dict = {}
    hook_handles = []
    try:
        for ln in target_cavs:
            h = model.get_submodule(ln).register_forward_hook(
                get_activation(ln, activation_dict)
            )
            hook_handles.append(h)

        model.eval()
        for ln, cav in target_cavs.items():
            layer_scores = []
            for batch in loader:
                imgs, _ = _extract_images_and_labels(batch)
                imgs = imgs.to(DEVICE)
                with torch.enable_grad():
                    outputs = model(imgs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    if ln not in activation_dict:
                        raise KeyError(f"Missing activation for layer '{ln}'.")
                    f_l = activation_dict[ln]
                    h_k = outputs[:, target_idx]
                    grad = torch.autograd.grad(
                        h_k.sum(), f_l,
                        retain_graph=False, allow_unused=True
                    )[0]
                    if grad is None:
                        layer_scores.append(
                            torch.zeros(imgs.size(0), dtype=torch.bool, device=DEVICE)
                        )
                    else:
                        grad_flat = grad.detach().view(grad.size(0), -1)
                        grad_norm = F.normalize(grad_flat, p=2, dim=1)
                        sensitivity = torch.sum(grad_norm * cav, dim=1)
                        layer_scores.append((sensitivity > 0).detach())
                        del grad, grad_flat, grad_norm, sensitivity
                activation_dict.clear()
                del imgs, outputs, h_k

            if layer_scores:
                tcav = torch.cat(layer_scores).float().mean().item()
                scores[ln] = tcav
                logging.info(
                    f"  TCAV layer={ln} target={target_idx} score={tcav:.4f}"
                )
                print(
                    f"  TCAV layer={ln} target={target_idx} score={tcav:.4f}"
                )
    finally:
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        activation_dict.clear()
    return scores


# ---------------------------------------------------------------------------
# Baseline metrics before recalibration
# ---------------------------------------------------------------------------

def _save_before_metrics(
    model,
    layer_names: list,
    registry: CAVRegistry,
    model_name: str,
    concept_name_map: dict,
    validation_loader,
    target_idx_list: list,
    results_path: str,
    logging,
):
    acc, prec, rec, f1, conf, class_res = _compute_metrics(
        model, validation_loader, target_idx_list, logging
    )
    logging.info(
        f"Before acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
    )
    print(f"Before acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    tcav_before = {}
    for target_idx in target_idx_list:
        concept_name = concept_name_map.get(target_idx)
        if concept_name is None:
            continue
        try:
            cavs = _load_target_cavs(
                registry, model_name, layer_names, concept_name, logging
            )
            if cavs:
                tcav_before[target_idx] = _compute_tcav_scores(
                    model, cavs, validation_loader, target_idx, logging
                )
            del cavs
        except Exception as exc:
            logging.warning(
                f"TCAV before-metrics failed target={target_idx}: {exc}"
            )

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    out_path = os.path.join(results_path, "accuracy_results_before.txt")
    with open(out_path, "w") as fh:
        fh.write("Accuracy Metrics Before Recalibration\n")
        fh.write("=" * 50 + "\n")
        fh.write(f"Accuracy : {acc:.4f}\n")
        fh.write(f"Precision: {prec:.4f}\n")
        fh.write(f"Recall   : {rec:.4f}\n")
        fh.write(f"F1 Score : {f1:.4f}\n")
        if isinstance(conf, dict):
            fh.write(f"Avg Conf : {conf}\n")
        if isinstance(class_res, dict):
            fh.write("\nClass-wise Results:\n")
            for cls_name, res in class_res.items():
                fh.write(
                    f"  {cls_name}: correct={res['correct']} total={res['total']}\n"
                )
        fh.write("\nTCAV Scores Before Recalibration:\n")
        for tidx, layer_scores in tcav_before.items():
            for ln, s in layer_scores.items():
                fh.write(f"  target={tidx} layer={ln}: {s:.4f}\n")
    logging.info(f"Before-metrics saved: {out_path}")
    print(f"Before-metrics saved: {out_path}")

# ---------------------------------------------------------------------------
# Single lambda training pass — one deep copy, explicit CUDA cleanup
# ---------------------------------------------------------------------------

def _run_one_lambda(
    base_model,
    base_model_name: str,
    layer_names: list,
    target_cavs: dict,
    training_loader,
    validation_loader,
    target_class: int,
    lambda_align: float,
    epochs: int,
    lr: float,
    results_path: str,
    model_save_name: str,
    save_plots: bool,
    logging,
) -> dict:
    lambda_cls = round(1.0 - lambda_align, 2)
    logging.info(f"  lambda_align={lambda_align} lambda_cls={lambda_cls}")
    print(f"  lambda_align={lambda_align} lambda_cls={lambda_cls}")

    model_trained = copy.deepcopy(base_model).to(DEVICE)
    activation_dict = {}
    hook_handles = []
    run_result = {}

    try:
        for ln in target_cavs:
            h = model_trained.get_submodule(ln).register_forward_hook(
                get_activation(ln, activation_dict)
            )
            hook_handles.append(h)

        for pname, param in model_trained.named_parameters():
            param.requires_grad = any(ln in pname for ln in layer_names)
        for module in model_trained.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
                module.eval()

        trainable_params = [p for p in model_trained.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(f"No trainable parameters for layers {layer_names}.")

        optimizer = optim.Adam(trainable_params, lr=lr)
        ce_loss_fn = nn.CrossEntropyLoss()
        loss_history = {"total": [], "cls": [], "align": []}
        model_trained.train()

        for epoch in range(epochs):
            total_l = cls_l_sum = align_l_sum = 0.0
            n_batches = 0

            for imgs, labels in training_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                outputs = model_trained(imgs)
                if isinstance(outputs, (tuple, list)):
                    main_out = outputs[0]
                    aux_out = outputs[1] if len(outputs) > 1 else None
                else:
                    main_out = outputs
                    aux_out = None

                if base_model_name == "inception_v3" and aux_out is not None:
                    cls_loss = (
                        0.6 * ce_loss_fn(main_out, labels)
                        + 0.4 * ce_loss_fn(aux_out, labels)
                    )
                else:
                    cls_loss = ce_loss_fn(main_out, labels)

                align_loss = torch.tensor(0.0, device=DEVICE)
                target_mask = labels == target_class

                if target_mask.any():
                    for ln, cav in target_cavs.items():
                        if ln not in activation_dict:
                            raise KeyError(f"Missing activation for '{ln}'.")
                        f_l = activation_dict[ln]
                        f_flat = F.normalize(
                            f_l.view(f_l.size(0), -1)[target_mask], p=2, dim=1
                        )
                        if f_flat.size(1) != cav.size(1):
                            raise ValueError(
                                f"CAV/feature dim mismatch at '{ln}': "
                                f"feat={f_flat.size(1)} cav={cav.size(1)}"
                            )
                        cosine_sim = torch.sum(f_flat * cav, dim=1)
                        align_loss = align_loss + (1 - cosine_sim.mean())
                        del f_l, f_flat, cosine_sim

                loss = lambda_align * align_loss + lambda_cls * cls_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=7.0)
                optimizer.step()

                total_l += loss.item()
                cls_l_sum += cls_loss.item()
                align_l_sum += align_loss.item()
                n_batches += 1

                # Free activation state and all batch tensors before next iteration
                activation_dict.clear()
                del imgs, labels, outputs, main_out, loss, cls_loss, align_loss
                if aux_out is not None:
                    del aux_out

            if n_batches > 0:
                loss_history["total"].append(total_l / n_batches)
                loss_history["cls"].append(cls_l_sum / n_batches)
                loss_history["align"].append(align_l_sum / n_batches)
                msg = (
                    f"  Epoch {epoch+1}/{epochs} "
                    f"total={loss_history['total'][-1]:.4f} "
                    f"align={loss_history['align'][-1]:.4f}"
                )
                logging.info(msg)
                print(msg)

        acc, prec, rec, f1, conf, class_res = _compute_metrics(
            model_trained, validation_loader, [target_class], logging
        )
        logging.info(
            f"  After: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
        )
        print(
            f"  After: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
        )

        run_result = {
            "Lambda Alignment": lambda_align,
            "Lambda Classification": lambda_cls,
            "Accuracy": round(float(acc), 6),
            "Precision": round(float(prec), 6),
            "Recall": round(float(rec), 6),
            "F1 Score": round(float(f1), 6),
        }
        if isinstance(conf, dict):
            run_result.update(
                {k: round(float(v), 6) if isinstance(v, (int, float)) else v
                 for k, v in conf.items()}
            )
        if isinstance(class_res, dict):
            for cid, cm in class_res.items():
                run_result[f"class{cid}_correct"] = cm.get("correct")
                run_result[f"class{cid}_total"] = cm.get("total")

        save_path = os.path.join(results_path, model_save_name)
        torch.save(model_trained, save_path)
        logging.info(f"  Saved model: {save_path}")

        if save_plots:
            layers_str = "_".join(layer_names)
            plot_loss_figure(
                loss_history["total"],
                loss_history["align"],
                loss_history["cls"],
                epochs,
                os.path.join(
                    results_path,
                    f"loss_cls_{base_model_name}_c{target_class}_{layers_str}_{lambda_align}.pdf",
                ),
                os.path.join(
                    results_path,
                    f"loss_align_{base_model_name}_c{target_class}_{layers_str}_{lambda_align}.pdf",
                ),
                os.path.join(
                    results_path,
                    f"loss_total_{base_model_name}_c{target_class}_{layers_str}_{lambda_align}.pdf",
                ),
            )

    except Exception as exc:
        logging.error(f"  Error at lambda={lambda_align}: {exc}")
        print(f"  Error at lambda={lambda_align}: {exc}")

    finally:
        # Always remove hooks and free model copy — prevents CUDA OOM across lambda iterations
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        activation_dict.clear()
        del model_trained
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return run_result


# ---------------------------------------------------------------------------
# Recalibrate one class across all lambda values for one layer combo
# ---------------------------------------------------------------------------

def _recalibrate_one_class(
    base_model,
    base_model_name: str,
    layer_combo: list,
    registry: CAVRegistry,
    concept_name: str,
    training_loader,
    validation_loader,
    target_class: int,
    lambda_aligns: list,
    combo_idx: int,
    epochs: int,
    lr: float,
    results_path: str,
    save_plots: bool,
    logging,
) -> list:
    logging.info(
        f"class={target_class} concept={concept_name} combo={layer_combo}"
    )
    target_cavs = _load_target_cavs(
        registry, base_model_name, layer_combo, concept_name, logging
    )
    if not target_cavs:
        logging.warning(
            f"No CAVs for class={target_class} concept={concept_name} "
            f"layers={layer_combo} — skipping."
        )
        return []

    run_metrics = []
    for lambda_idx, lambda_align in enumerate(lambda_aligns):
        set_seed(RANDOM_STATE + lambda_idx)
        result = _run_one_lambda(
            base_model=base_model,
            base_model_name=base_model_name,
            layer_names=layer_combo,
            target_cavs=target_cavs,
            training_loader=training_loader,
            validation_loader=validation_loader,
            target_class=target_class,
            lambda_align=lambda_align,
            epochs=epochs,
            lr=lr,
            results_path=results_path,
            model_save_name=(
                f"model_cls{target_class}_combo{combo_idx}"
                f"_lambda{lambda_align}.pth"
            ),
            save_plots=save_plots,
            logging=logging,
        )
        if result:
            result["Target Class"] = target_class
            result["Layers"] = str(layer_combo)
            result["Combination"] = "|".join(layer_combo)
            result["Layer Count"] = len(layer_combo)
            run_metrics.append(result)

    # Free CAV tensors once all lambdas for this combo are done
    del target_cavs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return run_metrics


# ---------------------------------------------------------------------------
# Layer combination / group resolution
# ---------------------------------------------------------------------------

def _resolve_layer_combos(
    layer_names: list,
    use_groups: bool,
    registry: CAVRegistry,
    model_name: str,
) -> list:
    if use_groups:
        groups = registry.list_groups()
        model_groups = [
            g for g in groups
            if registry._manifest["layer_groups"][g]["model"] == model_name
        ]
        if model_groups:
            combos = [
                registry._manifest["layer_groups"][g]["layers"]
                for g in model_groups
            ]
            print(
                f"Using {len(combos)} manifest groups for {model_name}: {model_groups}"
            )
            return combos
        print(
            f"No manifest groups for {model_name}; falling back to combinations."
        )

    combos = []
    for size in (1, 2, 3):
        if len(layer_names) >= size:
            combos.extend(
                [list(c) for c in combinations(layer_names, size)]
            )
    return combos


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    base_model,
    base_model_name: str,
    registry: CAVRegistry,
    concept_name_map: dict,
    layer_names: list,
    training_loader,
    validation_loader,
    target_idx_list: list,
    lambda_aligns: list,
    recalibrate_flags: list,
    epochs: int,
    lr: float,
    results_path: str,
    use_groups: bool,
    save_plots: bool,
    logging,
):
    set_seed(RANDOM_STATE)
    logging.info("Recalibration started.")

    _save_before_metrics(
        base_model, layer_names, registry, base_model_name,
        concept_name_map, validation_loader, target_idx_list,
        results_path, logging,
    )

    layer_combos = _resolve_layer_combos(
        layer_names, use_groups, registry, base_model_name
    )
    if not layer_combos:
        logging.error("No layer combinations available. Aborting.")
        print("No layer combinations available. Aborting.")
        return

    combo_index = {idx + 1: "|".join(c) for idx, c in enumerate(layer_combos)}
    with open(
        os.path.join(results_path, f"layer_combinations_{base_model_name}.txt"), "w"
    ) as fh:
        fh.write(str(combo_index))

    combo_csv = os.path.join(
        results_path, f"recalibration_combo_accuracy_{base_model_name}.csv"
    )
    all_results = []

    for class_idx, target_class in enumerate(target_idx_list):
        flag = (
            recalibrate_flags[class_idx]
            if class_idx < len(recalibrate_flags)
            else 1
        )
        if flag != 1:
            logging.info(
                f"Skipping class idx={class_idx} target={target_class} (flag=0)"
            )
            print(
                f"Skipping class idx={class_idx} target={target_class} (flag=0)"
            )
            continue

        concept_name = concept_name_map.get(target_class)
        if concept_name is None:
            logging.warning(
                f"No concept mapping for target_class={target_class} — skipping."
            )
            continue

        class_results = []
        for combo_idx, combo_layers in enumerate(layer_combos):
            logging.info(
                f"Class={target_class} combo {combo_idx+1}/{len(layer_combos)}: "
                f"{combo_layers}"
            )
            print(
                f"\nClass={target_class} combo {combo_idx+1}/{len(layer_combos)}: "
                f"{combo_layers}"
            )

            combo_results = _recalibrate_one_class(
                base_model=base_model,
                base_model_name=base_model_name,
                layer_combo=combo_layers,
                registry=registry,
                concept_name=concept_name,
                training_loader=training_loader,
                validation_loader=validation_loader,
                target_class=target_class,
                lambda_aligns=lambda_aligns,
                combo_idx=combo_idx,
                epochs=epochs,
                lr=lr,
                results_path=results_path,
                save_plots=save_plots,
                logging=logging,
            )

            for r in combo_results:
                r["Class Index"] = class_idx
                save_statistics(r, combo_csv)

            class_results.extend(combo_results)
            all_results.extend(combo_results)

        if class_results:
            best = max(class_results, key=lambda r: r.get("Accuracy", 0.0))
            logging.info(
                f"Best for class={target_class}: layers={best['Layers']} "
                f"lambda={best['Lambda Alignment']} acc={best['Accuracy']:.6f}"
            )
            print(
                f"Best for class={target_class}: layers={best['Layers']} "
                f"lambda={best['Lambda Alignment']} acc={best['Accuracy']:.6f}"
            )

    if all_results:
        best_overall = max(all_results, key=lambda r: r.get("Accuracy", 0.0))
        msg = (
            f"\nBest overall: class={best_overall['Target Class']} "
            f"layers={best_overall['Layers']} "
            f"lambda={best_overall['Lambda Alignment']} "
            f"acc={best_overall['Accuracy']:.6f}"
        )
        logging.info(msg)
        print(msg)

    logging.info("Recalibration complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CAV-guided model recalibration (loads CAVs from cav_store)."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name, e.g. vgg16")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Root directory containing model weight sub-folders")
    parser.add_argument("--cav_store", type=str, default="./cav_store",
                        help="CAV store root (contains manifest.json)")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to YAML config, e.g. config/biased/vgg16.yaml")
    parser.add_argument("--store_results", type=str, default="./results",
                        help="Base directory for results and logs")
    parser.add_argument("--use_layer_groups", action="store_true",
                        help="Use layer groups from manifest instead of combinations")
    parser.add_argument(
        "--recalibrate_classes", type=str, default=None,
        help="Comma-separated class indices to recalibrate, e.g. '0,2'. "
             "Defaults to all target classes.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    config = ConfigSingleton(args.config_file)
    set_seed(config.SEED)

    BASE_MODEL = args.model_name.strip().lower()
    MODEL_PATH = get_model_weight_path(BASE_MODEL, args.model_path.strip())

    RESULTS_PATH = os.path.join(args.store_results, BASE_MODEL)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    log_filename = os.path.join(
        RESULTS_PATH,
        f"audit_trail_{BASE_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    LOGGING = Logger_Singleton(log_filename)
    LOGGING.info("Script started.")

    registry = CAVRegistry(os.path.abspath(args.cav_store))

    IMAGE_SIZE = get_base_model_image_size(BASE_MODEL)
    LOGGING.info(f"Loading {BASE_MODEL} from {MODEL_PATH}")
    print(f"Loading {BASE_MODEL} from {MODEL_PATH} on {DEVICE}")

    if DEVICE == "cpu":
        MODEL = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    else:
        MODEL = torch.load(MODEL_PATH, map_location=DEVICE)
    MODEL.to(DEVICE)
    MODEL.eval()

    # Layer names — config override or all layers
    all_layers = get_model_layers(MODEL)
    if config.OVERRIDE_RECALIB:
        layer_map = {
            "vgg16":              config.VGG_RECALIB,
            "resnet50":           config.RESNET50_RECALIB,
            "inception_v3":       config.INCEPTION_V3_RECALIB,
            "mobilenet_v3_small": config.MOBILENET_V3_SMALL_RECALIB,
            "mobilenet_v3_large": config.MOBILENET_V3_LARGE_RECALIB,
        }
        LAYER_NAMES = layer_map.get(BASE_MODEL, all_layers[2:])
    else:
        LAYER_NAMES = all_layers[2:]
    LOGGING.info(f"Recalibration layers: {LAYER_NAMES}")
    print(f"Recalibration layers: {LAYER_NAMES}")

    TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    try:
        train_folders, valid_folders, class_names = get_class_folder_dicts(
            config.CLASSIFICATION_DATA_BASE_PATH
        )
        TARGET_IDX_LIST = [
            class_names.index(cls) for cls in config.TARGET_CLASS_LIST
        ]

        generator = torch.Generator()
        generator.manual_seed(RANDOM_STATE)

        TRAINING_LOADER = DataLoader(
            MultiClassImageDataset(train_folders, transform=TRANSFORM),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        VALIDATION_LOADER = DataLoader(
            MultiClassImageDataset(valid_folders, transform=TRANSFORM),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
        )
        LOGGING.info("Data loaded successfully.")
    except Exception as exc:
        LOGGING.error(f"Data preparation failed: {exc}")
        raise

    # Concept -> class mapping (positional: CONCEPT_NAMES[i] for TARGET_CLASS_LIST[i])
    CONCEPT_NAMES = [
        os.path.basename(p.rstrip("/\\"))
        for p in config.CONCEPT_FOLDER_LIST
    ]
    CONCEPT_NAME_MAP = {
        TARGET_IDX_LIST[i]: CONCEPT_NAMES[i]
        for i in range(min(len(TARGET_IDX_LIST), len(CONCEPT_NAMES)))
    }
    LOGGING.info(f"Concept-class map: {CONCEPT_NAME_MAP}")
    print(f"Concept-class map: {CONCEPT_NAME_MAP}")

    # Recalibrate flags
    if args.recalibrate_classes is not None:
        enabled = set(int(x) for x in args.recalibrate_classes.split(","))
        RECALIBRATE_FLAGS = [
            1 if i in enabled else 0
            for i in range(len(TARGET_IDX_LIST))
        ]
    else:
        RECALIBRATE_FLAGS = [1] * len(TARGET_IDX_LIST)

    LOGGING.info(f"Recalibrate flags: {RECALIBRATE_FLAGS}")

    # Normalise LAMBDA_ALIGNS to a list (config may return dict or list)
    _raw_lambdas = config.LAMBDA_ALIGNS
    LAMBDA_ALIGNS_LIST = (
        list(_raw_lambdas.values())
        if isinstance(_raw_lambdas, dict)
        else list(_raw_lambdas)
    )

    main(
        base_model=MODEL,
        base_model_name=BASE_MODEL,
        registry=registry,
        concept_name_map=CONCEPT_NAME_MAP,
        layer_names=LAYER_NAMES,
        training_loader=TRAINING_LOADER,
        validation_loader=VALIDATION_LOADER,
        target_idx_list=TARGET_IDX_LIST,
        lambda_aligns=LAMBDA_ALIGNS_LIST,
        recalibrate_flags=RECALIBRATE_FLAGS,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        results_path=RESULTS_PATH,
        use_groups=args.use_layer_groups,
        save_plots=getattr(config, "SAVE_PLOTS", False),
        logging=LOGGING,
    )

    LOGGING.info("Script execution finished.")
