import os
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}


def unique_name(dest_dir: Path, name: str) -> str:
    base, ext = os.path.splitext(name)
    candidate = name
    i = 1
    while (dest_dir / candidate).exists():
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate


def main():
    default_src = Path(r"D:\cuu_hidro\data\cam5_24H")
    default_dest = default_src / "all_images"

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else default_src
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else default_dest

    if not src.exists() or not src.is_dir():
        print(f"Source folder does not exist or is not a directory: {src}")
        sys.exit(1)

    # Avoid copying into a destination that is inside the source tree unless user asked explicitly
    try:
        src_resolved = src.resolve()
        dest_resolved = dest.resolve()
    except Exception:
        src_resolved = src
        dest_resolved = dest

    if src_resolved == dest_resolved:
        print("Source and destination are the same. Choose a different destination folder.")
        sys.exit(1)

    # Create dest
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    collisions = 0
    errors = 0

    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        # skip the destination folder if it's inside source
        try:
            if dest_resolved in root_path.resolve().parents or root_path.resolve() == dest_resolved:
                # This avoids copying files from the destination itself
                continue
        except Exception:
            pass

        for f in files:
            fp = root_path / f
            if fp.suffix.lower() not in IMAGE_EXTS:
                skipped += 1
                continue

            parent_name = root_path.name
            # Compose new name: parent_originalname
            new_name = f"{parent_name}_{f}" if parent_name else f
            dest_path = dest / new_name

            # If exists, use numeric suffix
            if dest_path.exists():
                collisions += 1
                new_name = unique_name(dest, os.path.splitext(new_name)[0] + fp.suffix)
                dest_path = dest / new_name

            try:
                shutil.copy2(fp, dest_path)
                copied += 1
            except Exception as e:
                print(f"Error copying {fp} -> {dest_path}: {e}")
                errors += 1

    print("\nDone.")
    print(f"Source: {src}")
    print(f"Destination: {dest}")
    print(f"Copied: {copied}")
    print(f"Skipped (non-images): {skipped}")
    print(f"Collisions handled: {collisions}")
    print(f"Errors: {errors}")


if __name__ == '__main__':
    main()
