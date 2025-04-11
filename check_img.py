import os
import re
from pathlib import Path

# Define the dataset directories.
base_dirs = [
    Path.home() / "Documents/dt_dataset/sf1t/dataset_local",
    Path.home() / "Documents/dt_dataset/sf2t/dataset_local",
    Path.home() / "Documents/dt_dataset/sf3t/dataset_local",
    Path.home() / "Documents/dt_dataset/sf4t/dataset_local",
]

# File lists required in each folder.
required_files_X = {"undeformed.png", "deformed.png"}
required_files_y = {"bounds.json", "csforce_local.png", "nforce_local.png",
                    "cnforce_local.png", "disp_local.png", "sforce_local.png"}

# Regular expressions to capture the pair number from folder names.
regex_X = re.compile(r'^X(\d+)$')
regex_y = re.compile(r'^y(\d+)$')

def check_dataset_folder(dataset_path: Path):
    """
    Checks the dataset folder to verify that each Xn/yn pair
    contains the required files. Instead of printing, collects
    any missing folders or files into a list.
    
    Returns:
        A list of dictionaries, each containing the dataset path,
        pair number, and a list of issues found.
    """
    missing_list = []
    pairs = {}

    # Look for subdirectories and match them to Xn or yn.
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            match_x = regex_X.match(folder.name)
            match_y = regex_y.match(folder.name)
            if match_x:
                num = match_x.group(1)
                pairs.setdefault(num, {})["X"] = folder
            elif match_y:
                num = match_y.group(1)
                pairs.setdefault(num, {})["y"] = folder

    # If no pairs were found, record this.
    if not pairs:
        missing_list.append({
            "dataset": str(dataset_path),
            "issue": "No valid Xn/yn pairs found."
        })
        return missing_list

    # Check each identified pair for missing folders/files.
    for num in sorted(pairs, key=lambda x: int(x)):
        issues = []
        pair = pairs[num]
        if "X" not in pair:
            issues.append(f"Missing folder: X{num}")
        if "y" not in pair:
            issues.append(f"Missing folder: y{num}")

        if "X" in pair:
            missing_files = [f for f in required_files_X if not (pair["X"] / f).exists()]
            if missing_files:
                issues.append(f"In folder X{num} missing: {', '.join(missing_files)}")
        if "y" in pair:
            missing_files = [f for f in required_files_y if not (pair["y"] / f).exists()]
            if missing_files:
                issues.append(f"In folder y{num} missing: {', '.join(missing_files)}")

        if issues:
            missing_list.append({
                "dataset": str(dataset_path),
                "pair": num,
                "issues": issues
            })

    return missing_list

def main():
    overall_missing = []
    for dataset_dir in base_dirs:
        if not dataset_dir.exists():
            overall_missing.append({
                "dataset": str(dataset_dir),
                "issue": "Dataset directory not found."
            })
        else:
            overall_missing.extend(check_dataset_folder(dataset_dir))
    return overall_missing

if __name__ == '__main__':
    missing_items = main()
    # Display the list of missing items.
    for item in missing_items:
        print(item)
