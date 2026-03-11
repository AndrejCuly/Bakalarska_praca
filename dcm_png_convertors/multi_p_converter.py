import os
import pydicom
import numpy as np
import time
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_single_file(path, output_folder):
    filename = os.path.basename(path)

    # Skip macOS system files
    if filename.startswith("._"):
        return f"Skipped (system file): {filename}"

    try:
        ds = pydicom.dcmread(path, force=True)

        # Fix missing TransferSyntax if necessary
        if not hasattr(ds.file_meta, "TransferSyntaxUID"):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Skip files with no pixel data
        if not hasattr(ds, "PixelData"):
            return f"Skipped (no pixel data): {filename}"

        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply DICOM windowing if present
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = np.atleast_1d(ds.WindowCenter)[0]
            ww = np.atleast_1d(ds.WindowWidth)[0]
            min_window = wc - (ww / 2)
            max_window = wc + (ww / 2)
            scaled = np.clip((pixel_array - min_window) * 255 / (max_window - min_window), 0, 255)
        else:
            # fallback linear normalization
            min_val = pixel_array.min()
            max_val = pixel_array.max()
            if max_val == min_val:
                return f"Skipped (flat image): {filename}"
            scaled = (pixel_array - min_val) * 255 / (max_val - min_val)

        scaled = scaled.astype(np.uint8)

        img = Image.fromarray(scaled)
        output_path = os.path.join(output_folder, filename.replace(".dcm", ".png"))
        img.save(output_path)

        return f"Converted: {filename}"

    except Exception as e:
        return f"Failed: {filename} → {e}"

def convert_dicom_to_png(input_folder, output_folder, max_workers=None):
    start_time = time.perf_counter()
    os.makedirs(output_folder, exist_ok=True)

    # Collect all DICOM files recursively
    dicom_files = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    print(f"Found {len(dicom_files)} DICOM files.")

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_single_file, f, output_folder): f for f in dicom_files}

        for future in as_completed(futures):
            print(future.result())

    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    convert_dicom_to_png(
        r"D:\CSAW\2021-204-1-1\data",
        r"C:\Users\culya\Desktop\bc_thesis_ssd2_output",
        max_workers=5  # None → uses all CPU cores
    )
