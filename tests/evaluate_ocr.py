import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from jiwer import cer, wer
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ID = "kazars24/trocr-base-handwritten-ru"


def normalize_path(value: str, project_root: Path) -> Path:
    """Resolve both Windows-style and POSIX-style paths from labels.csv."""
    raw = str(value).strip().strip('"').strip("'")
    # CSV may contain Windows separators even when the script runs on Linux.
    raw = raw.replace("\\", "/")
    path = Path(raw)

    if path.is_absolute() and path.exists():
        return path

    # First try relative to the project root.
    candidate = project_root / path
    if candidate.exists():
        return candidate

    # Handle paths that already include the project folder name.
    parts = path.parts
    if parts and parts[0] == project_root.name:
        candidate = project_root.joinpath(*parts[1:])
        if candidate.exists():
            return candidate

    return candidate


def normalize_text(text: str) -> str:
    """Metric normalization: lowercase and collapse repeated whitespace."""
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def load_model(model_id: str, device: torch.device):
    print(f"Loading model: {model_id}")
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def predict_batch(images, processor, model, device, max_new_tokens: int):
    pixel_values = processor(
        images=images,
        return_tensors="pt"
    ).pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        num_beams=1
    )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained TrOCR on a split from labels.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/labels.csv",
        help="Path to labels.csv (default: data/processed/labels.csv)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate (default: val)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Hugging Face model ID (default: {MODEL_ID})"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="outputs/ocr_eval")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find labels CSV: {csv_path}\n"
            "Place evaluate_ocr.py in the project root or pass --csv."
        )

    df = pd.read_csv(csv_path)
    required_columns = {"path", "text", "split"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {sorted(missing_columns)}")


    if args.split.lower() != "all":
        df = df[
        df["split"].astype(str).str.lower() == args.split.lower()
    ].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for split='{args.split}'. "
            f"Available splits: {df['split'].dropna().unique().tolist()}"
        )

    df["resolved_path"] = df["path"].apply(
        lambda p: normalize_path(p, project_root)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Rows in split '{args.split}': {len(df)}")

    processor, model = load_model(args.model, device)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []

    rows = list(df[["path", "text", "resolved_path"]].itertuples(index=False, name=None))

    for start in tqdm(range(0, len(rows), args.batch_size), desc="OCR inference"):
        batch_rows = rows[start:start + args.batch_size]
        batch_images = []
        valid_rows = []

        for original_path, ground_truth, image_path in batch_rows:
            try:
                with Image.open(image_path) as image:
                    batch_images.append(image.convert("RGB").copy())
                valid_rows.append((original_path, str(ground_truth), str(image_path)))
            except Exception as exc:
                skipped.append({
                    "path": str(original_path),
                    "resolved_path": str(image_path),
                    "error": str(exc)
                })

        if not batch_images:
            continue

        predictions = predict_batch(
            batch_images,
            processor,
            model,
            device,
            args.max_new_tokens
        )

        for (original_path, ground_truth, image_path), prediction in zip(
            valid_rows, predictions
        ):
            records.append({
                "path": original_path,
                "resolved_path": image_path,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "ground_truth_normalized": normalize_text(ground_truth),
                "prediction_normalized": normalize_text(prediction),
            })

    if not records:
        raise RuntimeError(
            "No images were successfully processed. Check paths in labels.csv."
        )

    results = pd.DataFrame(records)
    references = results["ground_truth_normalized"].tolist()
    hypotheses = results["prediction_normalized"].tolist()

    metrics = {
        "model": args.model,
        "split": args.split,
        "device": str(device),
        "total_rows_in_split": int(len(df)),
        "processed_images": int(len(results)),
        "skipped_images": int(len(skipped)),
        "CER": float(cer(references, hypotheses)),
        "WER": float(wer(references, hypotheses)),
        "exact_match_accuracy": float(
            (results["ground_truth_normalized"] ==
             results["prediction_normalized"]).mean()
        ),
    }

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    skipped_path = output_dir / "skipped_images.json"

    results.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Model:                 {metrics['model']}")
    print(f"Split:                 {metrics['split']}")
    print(f"Processed images:      {metrics['processed_images']}")
    print(f"Skipped images:        {metrics['skipped_images']}")
    print(f"CER:                   {metrics['CER']:.4f} "
          f"({metrics['CER'] * 100:.2f}%)")
    print(f"WER:                   {metrics['WER']:.4f} "
          f"({metrics['WER'] * 100:.2f}%)")
    print(f"Exact match accuracy:  "
          f"{metrics['exact_match_accuracy'] * 100:.2f}%")
    print("\nSaved files:")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics:     {metrics_path}")
    print(f"Skipped:     {skipped_path}")

    print("\n========== EXAMPLES ==========")
    for row in results.head(10).itertuples(index=False):
        print(f"\nImage: {row.path}")
        print(f"GT:    {row.ground_truth}")
        print(f"Pred:  {row.prediction}")


if __name__ == "__main__":
    main()
