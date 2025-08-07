import argparse
import os
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, Features, Sequence, Value


# This script creates a minimal Hugging Face dataset compatible with
# scripts/ecg_fm/inference.py.  It uses the provided deid_3.xml ECG file as
# a placeholder example.


def build_dummy_dataset(xml_filename: str) -> Dataset:
    """Return an in-memory HF dataset with a single row.

    The dataset schema matches what `inference.py` expects: a column named
    `ecg_paths` where each row is a *list* of ECG file paths (strings).
    """

    data: Dict[str, List] = {
        "patient_id": ["dummy_patient"],
        "ecg_paths": [[xml_filename]],  # Note the double brackets: list-of-str per row
    }

    features = Features(
        {
            "patient_id": Value("string"),
            "ecg_paths": Sequence(Value("string")),
        }
    )

    ds = Dataset.from_dict(data, features=features)
    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a dummy ECG dataset for inference testing.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dummy_ecg_dataset",
        help="Directory where the Hugging Face dataset will be saved (default: ./dummy_ecg_dataset)",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "deid_3.xml"),
        help="Path to the example ECG XML file (default: scripts/ecg_fm/deid_3.xml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    xml_path = Path(args.xml_path).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # Build dataset (uses *relative* path so inference can prepend --data_dir)
    # We store only the filename part so that users can pass --data_dir to
    # inference.py pointing to the directory containing the xml file.
    dataset = build_dummy_dataset(xml_path.name)

    # Save dataset to disk
    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"Dummy dataset saved to {output_dir}\n")
    print("You can test inference with, for example:\n")
    print(
        "python scripts/ecg_fm/inference.py --dataset {out} --data_dir {dd} --checkpoint path/to/ckpt.pth "
        "--output_dir /tmp/ecg_embeddings --mode 12lead".format(out=output_dir, dd=xml_path.parent)
    )


if __name__ == "__main__":
    main()
