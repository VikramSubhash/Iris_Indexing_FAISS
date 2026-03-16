import os
import re
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import iris
# Note: ensure_dir, iris, and other modules are assumed to be available in your environment.

# Takes the filename as input and parses out the User ID, Image Number, SessionNo/EyeSide
def parse_filename(filename, database="CASIAV1"):
    base = os.path.basename(filename)
    name, extension = os.path.splitext(base)
    
    # Initialize all optional fields
    user_id = "none"
    session_number = "none"
    eye_side = "none"
    image_number = "none"

    db = database.upper()

    if db == "CASIAV1":
        # CASIA V1 format: {3_digit_user_id}_{Session_id}_{image_num}.jpg
        # Example: 001_1_01.jpg
        match = re.match(r"(\d{3})_(\d)_(\d+)", name)
        
        if not match:
            raise ValueError(f"Invalid CASIAV1 filename format: {filename}")

        user_id, session_number, image_number = match.groups()
        eye_side = "none"

    elif db in ("CASIA-IRIS-INTERVAL", "CASIA_IRIS_INTERVAL"):
        # CASIA-Iris-Interval format: S1{3_digit_user_id}{eye_side}{image_no}.jpg
        # Example: S1001L01.jpg  (S1 + 001 + L + 01)
        match = re.match(r"S1(\d{3})([LR])(\d{2})", name, re.IGNORECASE)
        
        if not match:
            raise ValueError(f"Invalid CASIA-Iris-Interval filename format: {filename}")

        user_id, eye_side, image_number = match.groups()
        session_number = "1"

    elif db in ("CASIA-THOUSAND", "CASIA_THOUSAND", "CASIA IRIS THOUSAND", "CASIA-IRIS-THOUSAND"):
        # CASIA Iris Thousand format (example path): .../002/R/S5002R00.jpg
        # Filename pattern: S5{3_digit_user_id}{eye_side}{image_no}.jpg
        # Example: S5002R00.jpg -> S5 + 002 + R + 00
        match = re.match(r"S5(\d{3})([LR])(\d{2})", name, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid CASIA-Thousand filename format: {filename}")
        user_id, eye_side, image_number = match.groups()
        session_number = "1"  # CASIA Thousand doesn't use session ids in filename

    elif db in ("IRIS-LAMP", "IRIS_LAMP"):
        # Iris-Lamp format: S2{3_digit_user_id}{eye_side}{image_no}.jpg
        # Example: S2001L01.jpg -> S2 + 001 + L + 01
        match = re.match(r"S2(\d{3})([LR])(\d{2})", name, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid Iris-Lamp filename format: {filename}")
        user_id, eye_side, image_number = match.groups()
        session_number = "1"  # Iris-Lamp doesn't use session ids in filename

    elif db == "IITD":
        # IITD format: {image_no}_{eyeside}.bmp (Filename only)
        # Directory structure: ./IITD/{3_digit_user_id}/...
        # Example Filename: 1_L.bmp or 10_R.bmp
        match = re.match(r"(\d{1,2})_([LR])", name, re.IGNORECASE)
        
        if not match:
            raise ValueError(f"Invalid IITD filename format: {filename}")
        
        image_number, eye_side = match.groups()
        # User ID and Session Number must be extracted from the path in scan_files
        session_number = "1" # Assuming a single session for initial processing

    else:
        raise ValueError(f"Unknown database specified: {database}")

    return {
        "user_id": user_id,
        "session_number": session_number,
        "image_number": image_number,
        "eye_side": eye_side.upper() # This returns 'L' or 'R'
    }

# Takes the directory path as input
# Returns a list of file metadata dicts and a sorted list of user_ids
def scan_files(path="./CASIA1", database="CASIAV1"):
    allfiles = []
    user_ids = set()
    database_upper = database.upper()

    for root, dirs, files in os.walk(path):
        # Path-based USER ID extraction for datasets that don't carry user id in filename
        current_user_id = "none"

        # For IITD: path looks like .../IITD/{user_id}/...
        if database_upper == "IITD":
            directory_name = os.path.basename(root) 
            if directory_name.isdigit() and len(directory_name) <= 3:
                current_user_id = directory_name

        # For CASIA Thousand / Iris-Lamp: files are in .../{user_id}/{L_or_R}/
        # e.g. /.../002/R/S5002R00.jpg -> here root ends with '.../002/R' so parent dir is user_id
        if database_upper in ("CASIA-THOUSAND", "CASIA_THOUSAND", "CASIA IRIS THOUSAND", "CASIA-IRIS-THOUSAND", "IRIS-LAMP", "IRIS_LAMP"):
            parent_dir = os.path.basename(os.path.dirname(root))  # this should return '002' if root is .../002/R
            if parent_dir.isdigit() and len(parent_dir) == 3:
                current_user_id = parent_dir

        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                full_path = os.path.join(root, f)
                try:
                    meta = parse_filename(f, database) 
                    
                    # Insert the User ID extracted from the path for IITD, CASIA-THOUSAND, and IRIS-LAMP
                    if database_upper in ("IITD", "CASIA-THOUSAND", "CASIA_THOUSAND", "CASIA IRIS THOUSAND", "CASIA-IRIS-THOUSAND", "IRIS-LAMP", "IRIS_LAMP") and current_user_id != "none":
                        meta["user_id"] = current_user_id

                    # Ensure we have a valid user ID before proceeding
                    if meta["user_id"] == "none":
                        # Skip files in directories that don't match the user_id pattern
                        # This behavior preserves the original code's skip behavior
                        continue 
                        
                    meta["filepath"] = full_path
                    allfiles.append(meta)
                    user_ids.add(meta["user_id"])
                except ValueError as e:
                    print(f"Skipping {full_path} (File: {f}): {e}") 
    return allfiles, sorted(list(user_ids))


# Example tests for parse_filename with the new dataset added
if __name__ == "__main__":
    file_metadata_casia_v1 = parse_filename("001_1_1.jpg", database="CASIAV1")
    print("CASIA V1:", file_metadata_casia_v1)

    file_metadata_casia_interval = parse_filename("S1001L01.jpg", database="CASIA-IRIS-INTERVAL")
    print("CASIA Interval:", file_metadata_casia_interval)

    file_metadata_iitd = parse_filename("01_R.bmp", database="IITD")
    print("IITD:", file_metadata_iitd)

    # CASIA Thousand example filename (S5 + userid 002 + R + 00)
    file_metadata_casia_thousand = parse_filename("S5002R00.jpg", database="CASIA-THOUSAND")
    print("CASIA Thousand:", file_metadata_casia_thousand)

    # Iris-Lamp example filename (S2 + userid 001 + L + 01)
    file_metadata_iris_lamp = parse_filename("S2001L01.jpg", database="IRIS-LAMP")
    print("Iris-Lamp:", file_metadata_iris_lamp)


# -------------------------
# pipeline function (FIXED)
# -------------------------
def pipeline(dataset_path = "./CASIA1", save_visuals = False, save_intermediates = True):

    # Determine database type from dataset_path
    database_name = "CASIAV1"
    ds_lower = dataset_path.lower()
    
    if "casia-iris-interval" in ds_lower or "casia_iris_interval" in ds_lower or "interval" in ds_lower:
        database_name = "CASIA-IRIS-INTERVAL"
    elif "iitd" in ds_lower:
        database_name = "IITD"
    elif "thousand" in ds_lower or "casia-thousand" in ds_lower or "casia_thousand" in ds_lower or "casia iris thousand" in ds_lower:
        database_name = "CASIA-THOUSAND"
    elif "iris-lamp" in ds_lower or "iris_lamp" in ds_lower or "lamp" in ds_lower:
        database_name = "IRIS-LAMP"
    else:
        # default remains CASIAV1
        database_name = "CASIAV1"
        
    ## Main Pipeline to run the model on all images
    print(f"Scanning Dataset {dataset_path} (Database: {database_name})")
    # scan_files now requires the database_name argument
    files, user_ids = scan_files(dataset_path, database=database_name)
    print(f"Found {len(files)} files for {len(user_ids)} unique users")

    if len(files) == 0:
        print("No files found - exiting pipeline.")
        return

    # init debug env
    print("Initialising Iris Pipeline")
    iris_pipeline = iris.IRISPipeline(env=iris.IRISPipeline.DEBUGGING_ENVIRONMENT)

    # init visualizer
    print("Initialising Iris Visualizer")
    iris_visualizer = iris.visualisation.IRISVisualizer()

    # Step 3: Prepare output dirs (Optional but good practice to use unique output dir)
    output_vis_dir = os.path.join(f"{dataset_path}_", "outputs_vis")
    seg_dir = os.path.join(output_vis_dir, "segmentation")
    norm_dir = os.path.join(output_vis_dir, "normalized")
    code_dir = os.path.join(output_vis_dir, "codes")

    output_npz_dir = os.path.join(f"{dataset_path}_", "outputs_npz")
    temp_dir = os.path.join(output_npz_dir, "templates")
    seg_res_dir = os.path.join(output_npz_dir, "segmentation")
    norm_res_dir = os.path.join(output_npz_dir, "normalized")
    
    # Ensure folders exists (assuming ensure_dir is defined elsewhere)
    if save_visuals:
        ensure_dir(seg_dir)
        ensure_dir(norm_dir)
        ensure_dir(code_dir)

    if save_intermediates:
        ensure_dir(temp_dir)
        ensure_dir(seg_res_dir)
        ensure_dir(norm_res_dir)

    print(f"Fields: {files[0].keys()}")
    for file_meta in tqdm(files, desc="Processing iris images"):
        img_path = file_meta["filepath"]
        
        # Create a unique name from extracted metadata
        name_parts = [file_meta.get("user_id", "none"), file_meta.get("session_number", "none"),
                      file_meta.get("eye_side", "none"), file_meta.get("image_number", "none")]
        # Clean up the name, remove default/redundant parts
        unique_name = "_".join([p for p in name_parts if p not in ["none", "1", None] and p])

        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"Could not read Image {img_path}")
                
            # Use file_meta['eye_side'] and 'unique_name'
            # The .get() provides 'R' as a default if eye_side is missing, then .lower() makes it 'r'
            current_eye_side = file_meta.get("eye_side", "R").lower()
            
            # *** THIS IS THE FIX ***
            # Convert 'l' to 'left' and 'r' to 'right'
            if current_eye_side == "l":
                current_eye_side = "left"
            else:
                # Corrected 'current_eys_side' to 'current_eye_side'
                current_eye_side = "right"
            
            iris_image = iris.IRImage(
                img_data=image, 
                image_id=unique_name, 
                eye_side=current_eye_side # This will now be 'left' or 'right'
            )
            output = iris_pipeline(iris_image)
            
            # Paths for saving
            seg_path = os.path.join(seg_dir, f"{unique_name}_segmentation.jpg")
            norm_path = os.path.join(norm_dir, f"{unique_name}_normalized.jpg")
            temp_path = os.path.join(temp_dir, f"{unique_name}_template.npz")
            code_path = os.path.join(code_dir, f"{unique_name}_code.jpg")

            # Save intermediate arrays (if requested)
            if save_intermediates:
                try:
                    seg_arr = iris_pipeline.call_trace.get('segmentation', None)
                    if seg_arr is not None:
                        seg_npz_path = os.path.join(seg_res_dir, f"{unique_name}_segmentation.npz")
                        np.savez_compressed(seg_npz_path, segmentation=seg_arr)

                    norm_arr = iris_pipeline.call_trace.get('normalization', None)
                    if norm_arr is not None:
                        norm_npz_path = os.path.join(norm_res_dir, f"{unique_name}_normalized.npz")
                        np.savez_compressed(norm_npz_path, normalization=norm_arr)
                except Exception as e:
                    print(f"Warning: failed to save intermediates for {img_path}: {e}")

            if save_visuals:
                # Save segmentation result
                canvas = iris_visualizer.plot_segmentation_map(
                    ir_image=iris.IRImage(img_data=image, eye_side=current_eye_side),
                    segmap=iris_pipeline.call_trace.get('segmentation', None),
                )
                plt.savefig(seg_path, bbox_inches='tight')
                plt.close('all')
                # Save normalization result
                canvas = iris_visualizer.plot_normalized_iris(
                    normalized_iris=iris_pipeline.call_trace.get('normalization', None),
                )
                plt.savefig(norm_path, bbox_inches='tight')
                plt.close('all')

                # Save printed Templates
                canvas = iris_visualizer.plot_iris_template(
                    iris_template=iris_pipeline.call_trace.get('encoder', None),
                )
                plt.savefig(code_path, bbox_inches='tight')
                plt.close('all')

            # Save templates as npz (if present in output AND if save_intermediates is on)
            if save_intermediates:
                iris_code = output.get('iris_template', None)
                if iris_code is not None:
                    # adjust indexing depending on your iris_template structure
                    try:
                        iris_codes = iris_code.iris_codes
                        mask_code = iris_codes[1] if len(iris_codes) > 1 else None
                        iris_code_arr = iris_codes[0] if len(iris_codes) > 0 else None
                        np.savez_compressed(temp_path, iris_code = iris_code_arr, mask_code = mask_code)
                    except Exception:
                        # fallback: try direct keys if different structure
                        try:
                            np.savez_compressed(temp_path, iris_template=iris_code)
                        except Exception as e:
                            print(f"Warning: failed to save template npz for {img_path}: {e}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # Change this to your actual dataset folder.
    # Expected structure for Iris-Lamp:
    # DATASET_PATH/
    #   001/
    #     L/
    #       S2001L01.jpg ... S2001L20.jpg
    #     R/
    #       S2001R01.jpg ...
    #   002/
    #     L/
    #     R/
    #   ...
    #   411/
    
    # Example path for CASIA-Thousand:
    DATASET_PATH = "/home/nishkal/sg/iris_indexing/CASIA-Iris-Thousand"
    
    # Example path for Iris-Lamp:
    # DATASET_PATH = "/home/nishkal/sg/iris_indexing/CASIA-Iris-Lamp" # <<--- SET THIS TO YOUR DATASET ROOT
    pipeline(dataset_path=DATASET_PATH, save_visuals=True, save_intermediates=True)
    # Run the pipeline (set save_visuals/save_intermediates as needed)
    # if DATASET_PATH == "/path/to/your/Iris-Lamp" or DATASET_PATH == "/home/nishkal/sg/iris_indexing/CASIA-Iris-Lamp":
    #      print(f"Running pipeline for: {DATASET_PATH}")
         
    # else:
    #     print("Please update DATASET_PATH in the main block to point to your dataset.")