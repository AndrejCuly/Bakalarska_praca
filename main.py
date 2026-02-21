import os
import pydicom
import numpy as np
import time
from PIL import Image

def convert_dicom_to_png(input_folder, output_folder):

    start_time = time.perf_counter()

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        if not filename.lower().endswith(".dcm"):
            continue

        if filename.startswith("._"):
            continue

        if filename.lower().endswith(".dcm"):
            path = os.path.join(input_folder, filename)

            try:
                ds = pydicom.dcmread(path, force=True)

                # Fix missing TransferSyntax if necessary
                if not hasattr(ds.file_meta, "TransferSyntaxUID"):
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

                pixel_array = ds.pixel_array.astype(float)

                # Normalize to 0–255
                scaled = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
                scaled = np.uint8(scaled)

                img = Image.fromarray(scaled)
                output_path = os.path.join(output_folder, filename.replace(".dcm", ".png"))
                img.save(output_path)

                print(f"Converted: {filename}")

            except Exception as e:
                print(f"Failed: {filename} → {e}")


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    convert_dicom_to_png(
        r"C:\Users\culya\Desktop\test",
        r"C:\Users\culya\Desktop\bc_thesis_ssd2_output"
    )
