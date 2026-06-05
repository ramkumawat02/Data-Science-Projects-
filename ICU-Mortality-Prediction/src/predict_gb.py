import argparse
import pickle
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_PIPELINE_PATH = Path("models/gb_model.pkl")
DEFAULT_BUNDLE_PATH = Path("models/gb_shap_inference_bundle.pkl")
DEFAULT_DATA_PATH = Path("data/processed/icu_features.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved gradient boosting artifact and print sample predictions."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_PIPELINE_PATH,
        help="Path to a saved model artifact (.pkl).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV used to source a sample row.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index from the CSV to score.",
    )
    return parser.parse_args()


def load_bundle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file)


def load_sample_row(path: Path, row_index: int, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row {row_index} is out of bounds for dataset of size {len(df)}")
    return df.loc[[row_index], features]


def predict_with_pipeline(artifact_path: Path, sample: pd.DataFrame) -> None:
    model = joblib.load(artifact_path)
    probs = model.predict_proba(sample)[:, 1]
    preds = model.predict(sample)

    print(f"artifact: {artifact_path}")
    print(f"type: {type(model)}")
    print("sample:")
    print(sample.to_string(index=False))
    print(f"predicted_probability: {float(probs[0]):.6f}")
    print(f"predicted_class: {int(preds[0])}")


def predict_with_bundle(artifact_path: Path, sample: pd.DataFrame) -> None:
    bundle = load_bundle(artifact_path)
    imputer = bundle["imputer"]
    model = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle["threshold"])

    sample_imp = imputer.transform(sample[features])
    sample_imp_df = pd.DataFrame(sample_imp, columns=features, index=sample.index)
    prob = float(model.predict_proba(sample_imp_df)[0, 1])
    pred = int(prob >= threshold)

    print(f"artifact: {artifact_path}")
    print(f"type: {type(bundle)}")
    print(f"threshold: {threshold}")
    print("sample:")
    print(sample.to_string(index=False))
    print(f"predicted_probability: {prob:.6f}")
    print(f"predicted_class: {pred}")


def main() -> None:
    args = parse_args()

    if args.artifact.name == DEFAULT_BUNDLE_PATH.name:
        bundle = load_bundle(args.artifact)
        features = bundle["features"]
        sample = load_sample_row(args.data, args.row, features)
        predict_with_bundle(args.artifact, sample)
        return

    model = joblib.load(args.artifact)
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise ValueError(f"Could not infer features from artifact: {args.artifact}")

    sample = load_sample_row(args.data, args.row, features)
    predict_with_pipeline(args.artifact, sample)


if __name__ == "__main__":
    main()
