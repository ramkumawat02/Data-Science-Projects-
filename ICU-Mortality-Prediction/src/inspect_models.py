import argparse
import pickle
import warnings
import zipfile
from pathlib import Path

import joblib
from sklearn.exceptions import InconsistentVersionWarning


DEFAULT_MODEL_PATHS = [
    Path("models/gb_model.pkl"),
    Path("models/gradient_boosting.pkl"),
    Path("models/gb_shap_inference_bundle.pkl"),
    Path("models/lstm_model.pt"),
]


def print_header(path: Path) -> None:
    print(f"\n== {path} ==")


def print_kv(title: str, value) -> None:
    print(f"{title}: {value}")


def safe_joblib_load(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(path)


def summarize_estimator_params(estimator) -> dict:
    if not hasattr(estimator, "get_params"):
        return {}

    params = estimator.get_params(deep=False)
    preferred = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "random_state",
        "subsample",
        "criterion",
        "loss",
        "strategy",
    ]
    return {key: params[key] for key in preferred if key in params}


def inspect_joblib_model(path: Path) -> None:
    obj = safe_joblib_load(path)
    print_kv("type", type(obj))

    if hasattr(obj, "named_steps"):
        print_kv("pipeline steps", list(obj.named_steps.keys()))

        for step_name, step in obj.named_steps.items():
            print_kv(f"{step_name} type", type(step))
            step_params = summarize_estimator_params(step)
            if step_params:
                print_kv(f"{step_name} params", step_params)

    if hasattr(obj, "feature_names_in_"):
        print_kv("feature_names_in_", list(obj.feature_names_in_))

    if hasattr(obj, "get_params"):
        params = obj.get_params(deep=False)
        print_kv("top-level params", sorted(params.keys()))

    estimator_params = summarize_estimator_params(obj)
    if estimator_params:
        print_kv("estimator params", estimator_params)


def inspect_pickle_bundle(path: Path) -> None:
    with path.open("rb") as file:
        obj = pickle.load(file)

    print_kv("type", type(obj))

    if isinstance(obj, dict):
        print_kv("keys", list(obj.keys()))

        features = obj.get("features")
        if features is not None:
            print_kv("feature count", len(features))
            print_kv("features", features)

        threshold = obj.get("threshold")
        if threshold is not None:
            print_kv("threshold", threshold)

        seed = obj.get("seed")
        if seed is not None:
            print_kv("seed", seed)

        split_mode = obj.get("split_mode")
        if split_mode is not None:
            print_kv("split_mode", split_mode)

        notes = obj.get("notes")
        if notes is not None:
            print_kv("notes", notes)

        model = obj.get("model")
        if model is not None:
            print_kv("model type", type(model))
            model_params = summarize_estimator_params(model)
            if model_params:
                print_kv("model params", model_params)

        imputer = obj.get("imputer")
        if imputer is not None:
            print_kv("imputer type", type(imputer))
            imputer_params = summarize_estimator_params(imputer)
            if imputer_params:
                print_kv("imputer params", imputer_params)


def inspect_torch_checkpoint(path: Path) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch: not installed in this environment")
        inspect_torch_archive(path)
        return

    obj = torch.load(path, map_location="cpu")
    print_kv("type", type(obj))

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print_kv(f"state_dict keys ({len(keys)})", keys)

        tensor_shapes = {
            key: tuple(value.shape)
            for key, value in obj.items()
            if hasattr(value, "shape")
        }
        if tensor_shapes:
            print("tensor shapes:")
            for key, shape in tensor_shapes.items():
                print(f"  {key}: {shape}")


def inspect_torch_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        print_kv(f"zip members ({len(names)})", names)


def inspect_path(path: Path) -> None:
    print_header(path)

    if not path.exists():
        print("missing")
        return

    print_kv("size_bytes", path.stat().st_size)

    if path.suffix == ".pt":
        inspect_torch_checkpoint(path)
        return

    if path.name == "gb_shap_inference_bundle.pkl":
        inspect_pickle_bundle(path)
        return

    inspect_joblib_model(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect saved .pkl and .pt model artifacts."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_MODEL_PATHS,
        help="Model artifact paths to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.paths:
        inspect_path(path)


if __name__ == "__main__":
    main()
