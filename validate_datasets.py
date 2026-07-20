#!/usr/bin/env python
"""
Dataset Validation and Setup Script

Validates dataset structure, counts files, and provides setup instructions.
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

from scene_motion_llm.utils.config import (
    UCF101_PATH,
    KINETICS_PATH,
    CMU_MOSEI_PATH,
    LIRIS_ACCEDE_PATH
)

DATASETS = {
    "Kinetics": {
        "path": KINETICS_PATH,
        "type": "video",
        "description": "Large-Scale Action Recognition (400 classes)",
        "required": True
    },
    "LIRIS-ACCEDE": {
        "path": LIRIS_ACCEDE_PATH,
        "type": "video",
        "description": "Facial Expression and Sentiment Database",
        "required": True
    },
    "CMU-MOSEI": {
        "path": CMU_MOSEI_PATH,
        "type": "feature",
        "description": "Multimodal Sentiment Analysis (Optional)",
        "required": False
    }
}

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
FEATURE_EXTENSIONS = [".features", ".h5", ".hdf5"]

# ============================================
# DATASET VALIDATION
# ============================================

class DatasetValidator:
    """Validates dataset structure and contents"""

    def __init__(self):
        self.results = {}

    def validate_all(self):
        """Validate all datasets"""

        logger.info("=" * 80)
        logger.info("DATASET VALIDATION")
        logger.info("=" * 80)

        for dataset_name, dataset_info in DATASETS.items():
            self.validate_dataset(dataset_name, dataset_info)

        self.print_report()
        return self.results

    def validate_dataset(self, name, info):
        """Validate single dataset"""

        logger.info(f"\nValidating: {name}")

        dataset_path = Path(info["path"])
        result = {
            "name": name,
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "type": info["type"],
            "description": info["description"],
            "required": info["required"],
            "files": {},
            "issues": []
        }

        if not dataset_path.exists():
            result["issues"].append(f"Path does not exist: {dataset_path}")
            self.results[name] = result
            return

        # Count files by type
        if info["type"] == "video":
            result["files"] = self._count_video_files(dataset_path)
        elif info["type"] == "feature":
            result["files"] = self._count_feature_files(dataset_path)

        # Validate structure
        if info["type"] == "video":
            result["structure_valid"] = self._validate_video_structure(
                dataset_path, name
            )
        else:
            result["structure_valid"] = self._validate_feature_structure(
                dataset_path, name
            )

        # Check for empty directories
        if sum(result["files"].values()) == 0:
            result["issues"].append("No data files found")

        self.results[name] = result

    @staticmethod
    def _count_video_files(dataset_path):
        """Count video files by extension"""

        counts = defaultdict(int)

        for ext in VIDEO_EXTENSIONS:
            files = list(dataset_path.rglob(f"*{ext}"))
            counts[ext.replace(".", "")] = len(files)

        return dict(counts)

    @staticmethod
    def _count_feature_files(dataset_path):
        """Count feature files by extension"""

        counts = defaultdict(int)

        for ext in FEATURE_EXTENSIONS:
            files = list(dataset_path.rglob(f"*{ext}"))
            counts[ext.replace(".", "")] = len(files)

        return dict(counts)

    @staticmethod
    def _validate_video_structure(dataset_path, dataset_name):
        """Validate video dataset structure"""

        issues = []

        # Expected structures
        if dataset_name == "Kinetics":
            # Kinetics can have various structures, just check for video files
            video_files = list(dataset_path.rglob("*.mp4"))
            if len(video_files) == 0:
                issues.append("No video files found")

        elif dataset_name == "LIRIS-ACCEDE":
            for sentiment in ["positive", "neutral", "negative"]:
                sent_path = dataset_path / sentiment
                if not sent_path.exists():
                    issues.append(f"Missing sentiment folder: {sentiment}")
                else:
                    video_files = list(sent_path.rglob("*.mp4"))
                    if len(video_files) == 0:
                        issues.append(f"No video files in {sentiment} folder")

        return len(issues) == 0

    @staticmethod
    def _validate_feature_structure(dataset_path, dataset_name):
        """Validate feature dataset structure"""

        issues = []

        if dataset_name == "CMU-MOSEI":
            required_files = ["train.features", "test.features", "val.features"]
            for file in required_files:
                if not (dataset_path / file).exists():
                    issues.append(f"Missing file: {file}")

        return len(issues) == 0

    def print_report(self):
        """Print validation report"""

        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        total_valid = 0
        total_required = 0

        for name, result in self.results.items():
            status = "✓" if result["exists"] else "✗"
            required = "[REQUIRED]" if result["required"] else "[OPTIONAL]"

            print(f"\n{status} {name} {required}")
            print(f"   Path: {result['path']}")
            print(f"   Description: {result['description']}")

            if result["exists"]:
                total_valid += 1
                if result["files"]:
                    print(f"   Files:")
                    for ext, count in result["files"].items():
                        print(f"     - {ext}: {count}")

                if result["structure_valid"]:
                    print(f"   Structure: ✓ Valid")
                else:
                    print(f"   Structure: ✗ Invalid")

                if result["issues"]:
                    print(f"   Issues:")
                    for issue in result["issues"]:
                        print(f"     - {issue}")
            else:
                print(f"   Status: Not found")

            if result["required"]:
                total_required += 1

        print("\n" + "=" * 80)
        print(f"Valid: {total_valid}/{len(self.results)}")
        print(f"Required: {total_required}/{len([r for r in self.results.values() if r['required']])}")
        print("=" * 80)

        return total_valid == len(self.results)

    def get_json_report(self, output_path):
        """Save report as JSON"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Report saved to: {output_path}")


# ============================================
# SETUP INSTRUCTIONS
# ============================================

class SetupGuide:
    """Provides setup instructions"""

    @staticmethod
    def print_setup_instructions():
        """Print dataset setup instructions"""

        print("\n" + "=" * 80)
        print("DATASET SETUP INSTRUCTIONS")
        print("=" * 80)

        print("""
Kinetics-400 Dataset (Required)
──────────────────────────────────────────────────────────────────────────
1. Download from: https://deepmind.com/research/open-source/kinetics
2. Extract to: scene_motion_llm/dataset/Kinetics/
3. Expected structure:
   Kinetics/
   ├── video_class_1/
   │   ├── video1.mp4
   │   └── ...
   ├── video_class_2/
   └── ...

LIRIS-ACCEDE Dataset (Required)
──────────────────────────────────────────────────────────────────────────
1. Download from: http://liris.cnrs.fr/accede/
2. Extract to: scene_motion_llm/dataset/Liris_Accede/
3. Expected structure:
   Liris_Accede/
   ├── positive/
   │   ├── video1.mp4
   │   └── ...
   ├── neutral/
   │   ├── video2.mp4
   │   └── ...
   └── negative/
       ├── video3.mp4
       └── ...

CMU-MOSEI Dataset (Optional)
──────────────────────────────────────────────────────────────────────────
1. Download from: http://multicomp.cs.cmu.edu/resources/cmu-mosei/
2. Extract to: scene_motion_llm/dataset/CMU_MOSEI/
3. Expected structure:
   CMU_MOSEI/
   ├── train.features
   ├── test.features
   ├── val.features
   ├── w2v.vectors
   └── CMU-MOSEI/ (video files, optional)
        """)

    @staticmethod
    def print_quick_setup():
        """Print quick setup checklist"""

        print("\n" + "=" * 80)
        print("QUICK SETUP CHECKLIST")
        print("=" * 80)

        checklist = [
            ("Python 3.8+", "Check: python --version"),
            ("PyTorch", "Check: python -c 'import torch'"),
            ("OpenCV", "Check: python -c 'import cv2'"),
            ("Dataset: Kinetics-400", "Download and extract to dataset/Kinetics/"),
            ("Dataset: LIRIS-ACCEDE", "Download and extract to dataset/Liris_Accede/"),
            ("(Optional) CMU-MOSEI", "Download and extract to dataset/CMU_MOSEI/"),
            ("(Optional) GPU/CUDA", "For faster training (CPU works too)"),
        ]

        for i, (item, instruction) in enumerate(checklist, 1):
            print(f"{i}. {item}")
            print(f"   → {instruction}\n")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Validation and Setup Guide"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all datasets"
    )

    parser.add_argument(
        "--setup-guide",
        action="store_true",
        help="Show setup instructions"
    )

    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Quick setup checklist"
    )

    parser.add_argument(
        "--save-report",
        type=str,
        help="Save validation report as JSON"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks and show guides"
    )

    args = parser.parse_args()

    if not any([args.validate, args.setup_guide, args.quick_check, args.all]):
        parser.print_help()
        return

    if args.validate or args.all:
        validator = DatasetValidator()
        results = validator.validate_all()

        if args.save_report:
            validator.get_json_report(args.save_report)

    if args.setup_guide or args.all:
        SetupGuide.print_setup_instructions()

    if args.quick_check or args.all:
        SetupGuide.print_quick_setup()


if __name__ == "__main__":
    main()
