import argparse
import torch

import argparse
import pickle
import types
import torch

class _Placeholder:
    """Stand-in for classes that can't be imported (e.g. ActivationHook)."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
            self.__dict__.update(state[0])


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            return type(name, (_Placeholder,), {"__module__": module})


_safe_pickle = types.ModuleType("safe_pickle")
_safe_pickle.__dict__.update(pickle.__dict__)
_safe_pickle.Unpickler = _SafeUnpickler


def _extract_module_state(obj, prefix=""):
    """Rebuild a state_dict from an nn.Module-like object (even a placeholder)."""
    sd = {}
    d = getattr(obj, "__dict__", {})
    for name, t in (d.get("_parameters") or {}).items():
        if t is not None:
            sd[prefix + name] = t.detach() if torch.is_tensor(t) else t
    for name, t in (d.get("_buffers") or {}).items():
        if t is not None:
            sd[prefix + name] = t
    for name, child in (d.get("_modules") or {}).items():
        if child is not None:
            sd.update(_extract_module_state(child, prefix + name + "."))
    return sd


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", pickle_module=_safe_pickle, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        try:
            return obj.state_dict()
        except Exception:
            return _extract_module_state(obj)
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj:
                inner = obj[key]
                if isinstance(inner, dict):
                    return inner
                if hasattr(inner, "__dict__") and "_modules" in inner.__dict__:
                    return _extract_module_state(inner)
        return {k: v for k, v in obj.items() if torch.is_tensor(v)} or obj
    if hasattr(obj, "__dict__") and "_parameters" in obj.__dict__:
        return _extract_module_state(obj)
    raise ValueError(f"Unsupported checkpoint format: {path}")

def _print_diff_samples(k, a, b, matches, num_samples):
    diff_idx = (~matches).nonzero(as_tuple=False)[:num_samples]
    print(f"    Showing {len(diff_idx)} differing weights in '{k}':")
    print(f"    {'index':<25} {'model A':>16} {'model B':>16} {'abs diff':>14}")
    for idx in diff_idx:
        idx_t = tuple(idx.tolist())
        va = a[idx_t].float().item()
        vb = b[idx_t].float().item()
        print(f"    {str(idx_t):<25} {va:>16.8e} {vb:>16.8e} {abs(va - vb):>14.6e}")


def compare_models(sd_a, sd_b, atol=0.0, rtol=0.0,num_samples=15):
    all_same = True
    keys_a, keys_b = set(sd_a), set(sd_b)

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    for k in only_a:
        print(f"[MISSING in B] {k}")
        all_same = False
    for k in only_b:
        print(f"[MISSING in A] {k}")
        all_same = False

    for k in sorted(keys_a & keys_b):
        a, b = sd_a[k], sd_b[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            same = a == b
            print(f"[{'SAME' if same else 'DIFF'}] {k} (non-tensor)")
            all_same &= bool(same)
            continue
        if a.shape != b.shape:
            print(f"[SHAPE MISMATCH] {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
            all_same = False
            continue
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        if atol == 0.0 and rtol == 0.0:
            matches = a == b
        else:
            matches = torch.isclose(a.float(), b.float(), atol=atol, rtol=rtol)
        different_count = (~matches).sum().item()
        same = different_count == 0
        if same:
            print(f"[SAME] {k} (different weights: 0)")
        else:
            max_diff = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
            if num_samples > 0:
                _print_diff_samples(k, a, b, matches, num_samples)
            print(f"[DIFF] {k}: different weights = {different_count}, max abs diff = {max_diff:.6e}")
            all_same = False

    return all_same


def main():
    parser = argparse.ArgumentParser(description="Compare weights of two PyTorch models.")
    parser.add_argument("model_a", help="Path to model A checkpoint")
    parser.add_argument("model_b", help="Path to model B checkpoint")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance")
    args = parser.parse_args()

    sd_a = load_state_dict(args.model_a)
    sd_b = load_state_dict(args.model_b)

    if compare_models(sd_a, sd_b, args.atol, args.rtol):
        print("\nRESULT: Model A weights are the SAME as Model B.")
    else:
        print("\nRESULT: Model A weights are DIFFERENT from Model B.")


if __name__ == "__main__":
    main()

