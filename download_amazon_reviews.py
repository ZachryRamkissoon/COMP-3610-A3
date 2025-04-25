# Uncomment the lines below if you get a Hugging Face symlink warning on Windows
# ====================================================================================
# import os
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"
# ====================================================================================

import shutil
from pathlib import Path
import tarfile
from datasets import load_dataset, config, Dataset, DatasetDict, load_from_disk
from yaspin import yaspin
from typing import Optional, Union, List, Literal
import uuid

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="yaspin.core")

# Type alias for compression formats
CompressionFormat = Literal["gz", "bz2", "xz"]

def get_cache_directory(verbose: bool = True) -> Path:
    """Returns the current Hugging Face datasets cache directory as a Path object."""
    cache_dir = Path(config.HF_DATASETS_CACHE)
    if verbose:
        print(f"[INFO] Current cache directory: {cache_dir}")
        print(
            "[NOTE] To use a custom cache directory, set HF_DATASETS_CACHE before importing datasets.\n"
            "Example:\n"
            "    import os\n"
            "    os.environ['HF_DATASETS_CACHE'] = 'C:\\\\your\\\\custom\\\\path'\n"
            "    from datasets import load_dataset\n"
        )
    return cache_dir

def delete_cache_directory() -> None:
    """Deletes the Hugging Face datasets cache directory."""
    cache_path = Path(config.HF_DATASETS_CACHE)
    print(f"[INFO] Deleting Hugging Face cache at: {cache_path}")
    if cache_path.exists():
        shutil.rmtree(cache_path, ignore_errors=True)
        print("[SUCCESS] Cache directory deleted.")
    else:
        print(f"[WARNING] Cache directory does not exist: {cache_path}")

def default_cache_path() -> Path:
    """Returns and prints the default Hugging Face datasets cache path."""
    default_path = Path.home() / ".cache" / "huggingface" / "datasets"
    print(f'[INFO] Your default cache path: "{default_path}"')
    return default_path

# List of available categories
VALID_CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive",
    "Baby_Products", "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics",
    "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household",
    "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors",
    "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games", "Unknown"
]

def compress_folder(folder: Path, compression_format: CompressionFormat = "gz", level: int = 6) -> Path:
    """Compress a folder into a tar archive and delete the original folder."""
    if not 1 <= level <= 9:
        raise ValueError(f"Compression level must be between 1 and 9, got {level}")
    ext = ".tar.gz" if compression_format == "gz" else ".tar.bz2" if compression_format == "bz2" else ".tar.xz"
    mode = f"w:{compression_format}"
    archive_path = folder.with_suffix(ext)
    with tarfile.open(archive_path, mode, compresslevel=level if compression_format == "gz" else None) as tar:
        tar.add(folder, arcname=folder.name)
    print(f"[INFO] Using {compression_format.upper()} compression (level {level})")
    shutil.rmtree(folder)
    return archive_path

def process_dataset(dataset_type: str, category: str, base_save_path: Path, compress: bool, compression_format: CompressionFormat = "gz", compression_level: int = 6) -> str:
    """Download and save a specific dataset type for a category."""
    folder_name = f"raw_{dataset_type}_{category}"
    dataset_path = base_save_path / folder_name
    compressed_paths = [dataset_path.with_suffix(ext) for ext in [".tar.gz", ".tar.bz2", ".tar.xz"]]
    if dataset_path.exists() or any(path.exists() for path in compressed_paths):
        return f"[SKIP] {folder_name} already exists"
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_{dataset_type}_{category}", trust_remote_code=True)
    dataset_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dataset_path))
    if compress:
        compress_folder(dataset_path, compression_format=compression_format, level=compression_level)
        return f"[DONE] {folder_name} downloaded and compressed with {compression_format.upper()} level {compression_level}"
    return f"[DONE] {folder_name} downloaded"

def download_all_amazon_reviews(base_save_path: Union[str, Path], categories: Optional[List[str]] = None, compress: bool = False, compression_format: CompressionFormat = "gz", compression_level: int = 6) -> None:
    """Download Amazon review datasets for specified categories."""
    # Move cache to a separate E: drive location
    config.HF_DATASETS_CACHE = "E:/hf_cache"
    if categories is None:
        categories = VALID_CATEGORIES
    else:
        invalid = set(categories) - set(VALID_CATEGORIES)
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}")
    if not 1 <= compression_level <= 9:
        raise ValueError(f"Compression level must be between 1 and 9, got {compression_level}")
    if compression_format not in ["gz", "bz2", "xz"]:
        raise ValueError(f"Unsupported compression format: {compression_format}")
    hf_datasets_cache = get_cache_directory(verbose=False)
    base_save_path = Path(base_save_path).resolve()
    cache_path = Path(hf_datasets_cache).expanduser().resolve()
    if base_save_path == cache_path or base_save_path in cache_path.parents or cache_path in base_save_path.parents:
        raise ValueError("❌ base_save_path and HF_DATASETS_CACHE must be separate and non-overlapping.")
    base_save_path.mkdir(parents=True, exist_ok=True)
    successful = []
    failed = []
    if compress:
        print(f"[INFO] Using {compression_format.upper()} compression at level {compression_level}")
        print(f"[INFO] Compression speed: {'Fast' if compression_level < 4 else 'Medium' if compression_level < 7 else 'Slow'}")
        print(f"[INFO] Compression ratio: {'Low' if compression_level < 4 else 'Medium' if compression_level < 7 else 'High'}")
    for category in categories:
        with yaspin(text=f"Processing {category}") as spinner:
            try:
                review_result = process_dataset("review", category, base_save_path, compress, compression_format, compression_level)
                spinner.write(review_result)
                meta_result = process_dataset("meta", category, base_save_path, compress, compression_format, compression_level)
                spinner.write(meta_result)
                spinner.ok("✅")
                successful.append(category)
            except Exception as e:
                spinner.fail("💥")
                spinner.write(f"Failed to process category '{category}': {str(e)}")
                failed.append((category, str(e)))
            finally:
                if cache_path.exists():
                    shutil.rmtree(cache_path, ignore_errors=True)
    print(f"\n🎉 Download summary:")
    print(f"  - Successfully processed: {len(successful)}/{len(categories)} categories")
    if failed:
        print(f"  - Failed: {len(failed)}/{len(categories)} categories")
        for category, error in failed:
            print(f"    - {category}: {error}")

# Main execution to download all categories to E: drive, uncompressed
if __name__ == "__main__":
    save_path = Path("E:/Amazon_Reviews_2023")
    download_all_amazon_reviews(
        base_save_path=save_path,
        compress=False  
    )