import re
from pathlib import Path
from typing import Dict, List
from statistics import mean, stdev


def extract_inference_time(filepath: str) -> float:
    """
    Extract inference time from an info.txt or inference_time.txt file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Inference time in seconds, or None if not found
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract inference time from pattern like "Inference time (seed=23): 2.856 seconds"
            match = re.search(r'Inference time \(seed=\d+\):\s*([\d.]+)\s*seconds', content)
            if match:
                inference_time = float(match.group(1))
                return inference_time
    
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return None


def find_info_files(root_path: str) -> List[Path]:
    """
    Recursively find all info.txt and inference_time.txt files in a given directory.
    
    Args:
        root_path: Root directory path to search
        
    Returns:
        List of Path objects pointing to info.txt or inference_time.txt files
    """
    info_files = []
    root = Path(root_path)
    
    if not root.exists():
        print(f"Warning: Path does not exist: {root_path}")
        return info_files
    
    # Recursively find all info.txt and inference_time.txt files
    if root.is_dir():
        info_files.extend(root.rglob('info.txt'))
        info_files.extend(root.rglob('inference_time.txt'))
    elif root.is_file() and root.name in ['info.txt', 'inference_time.txt']:
        info_files.append(root)
    
    return info_files


def compute_mean_inference_times(directory_paths: List[str]) -> Dict[str, float]:
    """
    Compute mean inference times for each directory.
    
    Args:
        directory_paths: List of directory paths to analyze
        
    Returns:
        Dictionary mapping directory paths to their mean inference times and std devs in seconds
    """
    results = {}
    
    for dir_path in directory_paths:
        print(f"\n{dir_path}:")
        print("-" * 50)
        
        # Find all info.txt and inference_time.txt files in this directory
        info_files = find_info_files(dir_path)
        print(f"Found {len(info_files)} files (info.txt or inference_time.txt)")
        
        if not info_files:
            print(f"No files (info.txt or inference_time.txt) found")
            results[dir_path] = (None, None)
            continue
        
        # Extract inference times
        inference_times = []
        
        for filepath in info_files:
            inference_time = extract_inference_time(str(filepath))
            
            if inference_time is not None:
                inference_times.append(inference_time)
        
        # Compute mean and standard deviation
        if inference_times:
            mean_time = mean(inference_times)
            std_time = stdev(inference_times) if len(inference_times) > 1 else 0.0
            
            results[dir_path] = (mean_time, std_time)
            print(f"Valid inference times found: {len(inference_times)}")
            print(f"Mean inference time: {mean_time:.1f} ({std_time:.1f}) seconds")
        else:
            print(f"No valid inference times found")
            results[dir_path] = (None, None)
    
    return results


# Example usage
if __name__ == "__main__":
    
    MODELS_PATH = '../rq2-performance/'


    directory_paths = [
        MODELS_PATH + "slb/SLB-10/solutions/100k-10",
        MODELS_PATH + "slb/SLB-10/solutions/100k",
        MODELS_PATH + "slb/SLB-25/solutions/100k-10",
        MODELS_PATH + "slb/SLB-25/solutions/100k",
        MODELS_PATH + "slb/SLB-50/solutions/100k-10",
        MODELS_PATH + "slb/SLB-50/solutions/100k",
        MODELS_PATH + "slb/SLB-100/solutions/100k-10",
        MODELS_PATH + "slb/SLB-100/solutions/100k",
        MODELS_PATH + "cra/CRA-18/solutions/100k-10",
        MODELS_PATH + "cra/CRA-18/solutions/100k",
        MODELS_PATH + "cra/CRA-35/solutions/100k-10",
        MODELS_PATH + "cra/CRA-35/solutions/100k",
        MODELS_PATH + "cra/CRA-80/solutions/100k-10",
        MODELS_PATH + "cra/CRA-80/solutions/100k",
        MODELS_PATH + "cra/CRA-160/solutions/100k-10",
        MODELS_PATH + "cra/CRA-160/solutions/100k",
        MODELS_PATH + "sp/sprint-planning-model-4-stakeholders-42-items/solutions/10",
        MODELS_PATH + "sp/sprint-planning-model-4-stakeholders-42-items/solutions/conv",
        MODELS_PATH + "sp/sprint-planning-model-5-stakeholders-64-items/solutions/10",
        MODELS_PATH + "sp/sprint-planning-model-5-stakeholders-64-items/solutions/conv",
        MODELS_PATH + "sp/sprint-planning-model-5-stakeholders-119-items/solutions/10",
        MODELS_PATH + "sp/sprint-planning-model-5-stakeholders-119-items/solutions/conv",
        MODELS_PATH + "sp/sprint-planning-model-6-stakeholders-124-items/solutions/10",
        MODELS_PATH + "sp/sprint-planning-model-6-stakeholders-124-items/solutions/conv",
    ]
    
    print("Analyzing inference times per directory...\n")
    print("="*50)
    mean_times = compute_mean_inference_times(directory_paths)
    
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    for dir_path, (mean_time, std_time) in mean_times.items():
        if mean_time is not None:
            print(f"{dir_path}: {mean_time:.1f} ({std_time:.1f}) seconds")
        else:
            print(f"{dir_path}: No data")