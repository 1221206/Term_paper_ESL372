
import h5py
import os

def check_dimensions(mat_file):
    """Checks the dimensions of key datasets in a .mat file."""
    print(f"--- Checking dimensions in {os.path.basename(mat_file)} ---")
    try:
        with h5py.File(mat_file, 'r') as f:
            if 'batch' not in f:
                print("  'batch' group not found.")
                return

            batch_group = f['batch']
            print("  Shapes of datasets in 'batch' group:")
            for key, item in batch_group.items():
                if isinstance(item, h5py.Dataset):
                    print(f"    - {key}: {item.shape}")

            # --- Inspect the first battery as a sample ---
            print("\n  --- Inspecting first battery (index 0) ---")
            if batch_group['barcode'].shape[0] > 0:
                barcode_ref = batch_group['barcode'][0, 0]
                barcode_data = f[barcode_ref]
                print(f"    - Barcode: shape={barcode_data.shape}, dtype={barcode_data.dtype}")

                cycles_ref = batch_group['cycles'][0, 0]
                cycles_data = f[cycles_ref]
                print(f"    - Cycles: shape={cycles_data.shape}, dtype={cycles_data.dtype}")
                if cycles_data.dtype.names:
                    print(f"      Fields: {cycles_data.dtype.names}")
            else:
                print("    - No batteries found to inspect.")

    except Exception as e:
        print(f"  Could not read {os.path.basename(mat_file)}: {e}")

if __name__ == "__main__":
    mat_file_path = "C:\\Users\\pc\\Inf Hyp\\Power system\\Pi_KANN\\MIT_data\\2017-05-12_batchdata_updated_struct_errorcorrect.mat"
    check_dimensions(mat_file_path)
