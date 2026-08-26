from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import describe_path_resolution, train_label_path, val_label_path


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def import_required(name: str, failures: list[str]):
    try:
        module = importlib.import_module(name)
        print(f"PASS import {name}: {getattr(module, '__version__', 'ok')}")
        return module
    except Exception as exc:
        failures.append(f"import {name}: {type(exc).__name__}: {exc}")
        print(f"FAIL import {name}: {type(exc).__name__}: {exc}")
        return None


def readable_head(path: Path, byte_count: int = 4096) -> bool:
    with path.open("rb") as handle:
        handle.read(byte_count)
    return True


def check_checkpoint(path: Path, torch, failures: list[str]) -> None:
    check(path.exists(), f"checkpoint exists: {path}", failures)
    if not path.exists():
        return
    check(path.stat().st_size > 0, f"checkpoint size > 0: {path}", failures)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            keys = sorted(str(key) for key in payload.keys())
            print("PASS checkpoint readable keys: " + ", ".join(keys[:12]))
        else:
            print(f"PASS checkpoint readable object: {type(payload).__name__}")
    except Exception as exc:
        failures.append(f"checkpoint readable: {path}: {type(exc).__name__}: {exc}")
        print(f"FAIL checkpoint readable: {path}: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only RL-GenRisk migration healthcheck.")
    parser.add_argument("--train-label-path", default=None)
    parser.add_argument("--val-label-path", default=None)
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "data" / "agent_KIRC_driver_DDQN_PER.th"))
    args = parser.parse_args()

    failures: list[str] = []

    print(f"project_root: {PROJECT_ROOT}")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version}")
    print(describe_path_resolution())

    torch = import_required("torch", failures)
    torch_geometric = import_required("torch_geometric", failures)
    numpy = import_required("numpy", failures)
    pandas = import_required("pandas", failures)
    import_required("sklearn", failures)

    if torch is not None:
        print(f"torch_cuda_runtime: {torch.version.cuda}")
        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_available}")
        print(f"gpu_name: {torch.cuda.get_device_name(0) if cuda_available else 'NO_GPU'}")
        check(cuda_available, "CUDA is available", failures)
        if cuda_available:
            try:
                x = torch.tensor([1.0, 2.0], device="cuda")
                y = (x * 2).sum().item()
                check(y == 6.0, "CUDA tensor smoke", failures)
            except Exception as exc:
                failures.append(f"CUDA tensor smoke: {type(exc).__name__}: {exc}")
                print(f"FAIL CUDA tensor smoke: {type(exc).__name__}: {exc}")

    if torch is not None and torch_geometric is not None:
        try:
            from torch_geometric.nn import GCNConv

            device = "cuda" if torch.cuda.is_available() else "cpu"
            conv = GCNConv(2, 2).to(device)
            features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
            out = conv(features, edge_index)
            check(tuple(out.shape) == (2, 2), "GCNConv CUDA forward smoke", failures)
        except Exception as exc:
            failures.append(f"GCNConv smoke: {type(exc).__name__}: {exc}")
            print(f"FAIL GCNConv smoke: {type(exc).__name__}: {exc}")

    for module_name in ["qfunction", "replay_buffer", "DQN"]:
        import_required(module_name, failures)

    for path in [PROJECT_ROOT / "data", PROJECT_ROOT / "multi-omics data"]:
        check(path.exists() and path.is_dir(), f"directory exists: {path}", failures)

    labels = {
        "train_label": Path(args.train_label_path) if args.train_label_path else train_label_path(),
        "validation_label": Path(args.val_label_path) if args.val_label_path else val_label_path(),
    }
    for name, path in labels.items():
        lowered = str(path).lower()
        check("test_driver_genes" not in lowered and "external_holdout" not in lowered, f"{name} is not Test/external: {path}", failures)
        check(path.exists(), f"{name} exists: {path}", failures)
        if path.exists():
            try:
                readable_head(path)
                print(f"PASS {name} basic readable: {path}")
            except Exception as exc:
                failures.append(f"{name} readable: {type(exc).__name__}: {exc}")
                print(f"FAIL {name} readable: {type(exc).__name__}: {exc}")

    if pandas is not None:
        for path in [
            PROJECT_ROOT / "data" / "processed" / "KIRC_multiomics_3omics.csv",
            PROJECT_ROOT / "data" / "processed" / "KIRC_multiomics_4omics.csv",
        ]:
            check(path.exists(), f"data file exists: {path}", failures)
            if path.exists():
                try:
                    frame = pandas.read_csv(path, nrows=3)
                    print(f"PASS data readable: {path} columns={list(frame.columns)}")
                except Exception as exc:
                    failures.append(f"data readable {path}: {type(exc).__name__}: {exc}")
                    print(f"FAIL data readable {path}: {type(exc).__name__}: {exc}")

    if torch is not None:
        check_checkpoint(Path(args.checkpoint), torch, failures)

    print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
    if failures:
        print("PROJECT_HEALTHCHECK: FAIL")
        return 1
    print("PROJECT_HEALTHCHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
