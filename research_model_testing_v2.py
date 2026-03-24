import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

import shutil
import glob
import gc
import torch
from ultralytics import YOLO


# Constants for paths
BASE_HOME = "/home/c_srambo@lmumain.edu/Research_Project"
DATASET_PATH = os.path.join(BASE_HOME, "Research_Project_Data")
SAVE_DIR = os.path.join(DATASET_PATH, 'training_results')
WEIGHTS_PATH = os.path.join(BASE_HOME, 'yolo26m.pt')

def cleanup_environment():
    """Removes caches and frees up System RAM and GPU VRAM."""
    print("\n--- Deep Cleaning Environment ---")
    
    # 1. Target specific label.cache files in train, test, and valid folders
    subsets = ['train', 'test', 'valid', 'val']
    for subset in subsets:
        # This looks for labels.cache specifically in the labels subfolders
        pattern = os.path.join(DATASET_PATH, subset, "labels.cache")
        # This also looks for any .cache file anywhere in the dataset structure
        recursive_pattern = os.path.join(DATASET_PATH, subset, "**/*.cache")
        
        found_caches = glob.glob(pattern) + glob.glob(recursive_pattern, recursive=True)
        
        for f in set(found_caches):
            try:
                os.remove(f)
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")

    # 2. Delete local .pt files in current directory to prevent 'dirty' weights
    for weight_file in ['yolo26n.pt', 'yolo26m.pt']:
        if os.path.exists(weight_file):
            os.remove(weight_file)
            print(f"Removed temporary local weight: {weight_file}")

    # 3. Flush RAM and VRAM
    gc.collect() # System RAM
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache() # GPU VRAM
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"CUDA cleanup skipped or failed: {e}")


def run_experiment_batch(experiments):
    for exp in experiments:
        group_name = exp['name']
        
        # Reset all augmentations to 0.0 for every new group
        params = {
            'degrees': 0.0, 'hsv_v': 0.0, 'shear': 0.0, 'fliplr': 0.0,
            'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0, 'translate': 0.0,
            'scale': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'bgr': 0.0,
            'erasing': 0.0, 'cutmix': 0.0, 'hsv_h': 0.0, 'hsv_s': 0.0,
            'crop_fraction': 0.9 # Kept at 0.9 per your earlier script
        }
        
        # Apply your manual overrides
        params.update(exp['augments'])

        print(f"\n{'='*50}\nRUNNING EXPERIMENT: {group_name}\n{'='*50}")
        
        # Clean BEFORE training
        cleanup_environment()
        
        # Initialize and Train
        model = YOLO(WEIGHTS_PATH, task='detect')
        model.train(
            data=f'{DATASET_PATH}/data.yaml',
            epochs=200,
            imgsz=640,
            batch=96,
            workers=24,
            device=list(range(torch.cuda.device_count())),
            project=SAVE_DIR,
            name=group_name,
            exist_ok=True,
            amp=True,
            cache='ram',
            save=True,
            save_period=50,
            cls=1.5,
            box=7.5,
            dfl=1.5,
            **params # Passes the manual augmentations
        )

        # Force clean AFTER training to prepare for next loop
        del model
        cleanup_environment()

if __name__ == "__main__":
    # --- MANUAL CONFIGURATION AREA ---
    # Define your groups and their specific augmentations here
    my_experiments = [
        {"name": "Group_14_Augmentation_Set_0.2_Translate_12_Shear_0.5_Mosaic_140_Rotation", "augments": {"translate": 0.2, "shear": 12.0, "mosaic": 0.5, "degrees": 140.0}},
        {"name": "Group_15_Augmentation_Set_0.3_Translate_15_Shear_0.8_Mosaic_180_Rotation", "augments": {"translate": 0.3, "shear": 15.0, "mosaic": 0.8, "degrees": 180.0}},
        {"name": "Group_16_Augmentation_Set_0.4_Translate_0.0002_Perspective_1_Mosaic_0.1_Brightness", "augments": {"translate": 0.4, "perspective": 0.0002, "mosaic": 1.0, "hsv_v": 0.1}},
        {"name": "Group_17_Augmentation_Set_0.2_Scale_0.0005_Perspective_0.2_CutMix_0.3_Brightness", "augments": {"scale": 0.2, "perspective": 0.0005, "cutmix": 0.2, "hsv_v": 0.3}},
        {"name": "Group_18_Augmentation_Set_0.5_Scale_0.0008_Perspective_0.8_CutMix_0.5_Brightness", "augments": {"scale": 0.5, "perspective": 0.0008, "cutmix": 0.8, "hsv_v": 0.5}},
        {"name": "Group_19_Augmentation_Set_0.7_Scale_0.0010_Perspective_1_CutMix_0.8_Brightness", "augments": {"scale": 0.7, "perspective": 0.001, "cutmix": 1.0, "hsv_v": 0.8}},
        {"name": "Group_20_Augmentation_Set_0.9_Scale_0.2_Flip_LR_15_Rotation_1_Brightness", "augments": {"scale": 0.9, "fliplr": 0.2, "degrees": 15.0, "hsv_v": 1.0}},
        {"name": "Group_21_Augmentation_Set_3_Shear_0.5_Flip_LR_45_Rotation_0.1_Translate", "augments": {"shear": 3.0, "fliplr": 0.5, "degrees": 45.0, "translate": 0.1}},
        {"name": "Group_22_Augmentation_Set_6_Shear_0.8_Flip_LR_80_Rotation_0.2_Translate", "augments": {"shear": 6.0, "fliplr": 0.8, "degrees": 80.0, "translate": 0.2}},
        {"name": "Group_23_Augmentation_Set_9_Shear_0.2_Mosaic_110_Rotation_0.3_Translate", "augments": {"shear": 9.0, "mosaic": 0.2, "degrees": 110.0, "translate": 0.3}},
        {"name": "Group_24_Augmentation_Set_12_Shear_0.5_Mosaic_140_Rotation_0.4_Translate", "augments": {"shear": 12.0, "mosaic": 0.5, "degrees": 140.0, "translate": 0.4}},
        {"name": "Group_25_Augmentation_Set_15_Shear_0.8_Mosaic_180_Rotation_0.2_Scale", "augments": {"shear": 15.0, "mosaic": 0.8, "degrees": 180.0, "scale": 0.2}},
        {"name": "Group_26_Augmentation_Set_0.0002_Perspective_1_Mosaic_0.1_Brightness_0.5_Scale", "augments": {"perspective": 0.0002, "mosaic": 1.0, "hsv_v": 0.1, "scale": 0.5}},
        {"name": "Group_27_Augmentation_Set_0.0005_Perspective_0.2_CutMix_0.3_Brightness_0.7_Scale", "augments": {"perspective": 0.0005, "cutmix": 0.2, "hsv_v": 0.3, "scale": 0.7}},
        {"name": "Group_28_Augmentation_Set_0.0008_Perspective_0.8_CutMix_0.5_Brightness_0.9_Scale", "augments": {"perspective": 0.0008, "cutmix": 0.8, "hsv_v": 0.5, "scale": 0.9}},
        {"name": "Group_29_Augmentation_Set_0.0010_Perspective_1_CutMix_0.8_Brightness_3_Shear", "augments": {"perspective": 0.001, "cutmix": 1.0, "hsv_v": 0.8, "shear": 3.0}},
        {"name": "Group_30_Augmentation_Set_0.2_Flip_LR_15_Rotation_1_Brightness_6_Shear", "augments": {"fliplr": 0.2, "degrees": 15.0, "hsv_v": 1.0, "shear": 6.0}},
        {"name": "Group_31_Augmentation_Set_0.5_Flip_LR_45_Rotation_0.1_Translate_9_Shear", "augments": {"fliplr": 0.5, "degrees": 45.0, "translate": 0.1, "shear": 9.0}},
        {"name": "Group_32_Augmentation_Set_0.8_Flip_LR_80_Rotation_0.2_Translate_12_Shear", "augments": {"fliplr": 0.8, "degrees": 80.0, "translate": 0.2, "shear": 12.0}},
        {"name": "Group_33_Augmentation_Set_0.2_Mosaic_110_Rotation_0.3_Translate_15_Shear", "augments": {"mosaic": 0.2, "degrees": 110.0, "translate": 0.3, "shear": 15.0}},
        {"name": "Group_34_Augmentation_Set_0.5_Mosaic_140_Rotation_0.4_Translate_0.0002_Perspective", "augments": {"mosaic": 0.5, "degrees": 140.0, "translate": 0.4, "perspective": 0.0002}},
        {"name": "Group_35_Augmentation_Set_0.8_Mosaic_180_Rotation_0.2_Scale_0.0005_Perspective", "augments": {"mosaic": 0.8, "degrees": 180.0, "scale": 0.2, "perspective": 0.0005}},
        {"name": "Group_36_Augmentation_Set_1_Mosaic_0.1_Brightness_0.5_Scale_0.0008_Perspective", "augments": {"mosaic": 1.0, "hsv_v": 0.1, "scale": 0.5, "perspective": 0.0008}},
        {"name": "Group_37_Augmentation_Set_0.2_CutMix_0.3_Brightness_0.7_Scale_0.0010_Perspective", "augments": {"cutmix": 0.2, "hsv_v": 0.3, "scale": 0.7, "perspective": 0.001}},
        {"name": "Group_38_Augmentation_Set_0.8_CutMix_0.5_Brightness_0.9_Scale_0.2_Flip_LR", "augments": {"cutmix": 0.8, "hsv_v": 0.5, "scale": 0.9, "fliplr": 0.2}},
        {"name": "Group_39_Augmentation_Set_1_CutMix_0.8_Brightness_3_Shear_0.5_Flip_LR", "augments": {"cutmix": 1.0, "hsv_v": 0.8, "shear": 3.0, "fliplr": 0.5}},
        {"name": "Group_40_Augmentation_Set_15_Rotation_1_Brightness_6_Shear_0.8_Flip_LR", "augments": {"degrees": 15.0, "hsv_v": 1.0, "shear": 6.0, "fliplr": 0.8}},
        {"name": "Group_41_Augmentation_Set_45_Rotation_0.1_Translate_9_Shear_0.2_Mosaic", "augments": {"degrees": 45.0, "translate": 0.1, "shear": 9.0, "mosaic": 0.2}},
        {"name": "Group_42_Augmentation_Set_80_Rotation_0.2_Translate_12_Shear_0.5_Mosaic", "augments": {"degrees": 80.0, "translate": 0.2, "shear": 12.0, "mosaic": 0.5}},
        {"name": "Group_43_Augmentation_Set_110_Rotation_0.3_Translate_15_Shear_0.8_Mosaic", "augments": {"degrees": 110.0, "translate": 0.3, "shear": 15.0, "mosaic": 0.8}},
        {"name": "Group_44_Augmentation_Set_140_Rotation_0.4_Translate_0.0002_Perspective_1_Mosaic", "augments": {"degrees": 140.0, "translate": 0.4, "perspective": 0.0002, "mosaic": 1.0}},
        {"name": "Group_45_Augmentation_Set_180_Rotation_0.2_Scale_0.0005_Perspective_0.2_CutMix", "augments": {"degrees": 180.0, "scale": 0.2, "perspective": 0.0005, "cutmix": 0.2}},
        {"name": "Group_46_Augmentation_Set_0.1_Brightness_0.5_Scale_0.0008_Perspective_0.8_CutMix", "augments": {"hsv_v": 0.1, "scale": 0.5, "perspective": 0.0008, "cutmix": 0.8}},
        {"name": "Group_47_Augmentation_Set_0.3_Brightness_0.7_Scale_0.0010_Perspective_1_CutMix", "augments": {"hsv_v": 0.3, "scale": 0.7, "perspective": 0.001, "cutmix": 1.0}},
        {"name": "Group_48_Augmentation_Set_0.5_Brightness_0.9_Scale_0.2_Flip_LR_15_Rotation", "augments": {"hsv_v": 0.5, "scale": 0.9, "fliplr": 0.2, "degrees": 15.0}},
        {"name": "Group_49_Augmentation_Set_0.8_Brightness_3_Shear_0.5_Flip_LR_45_Rotation", "augments": {"hsv_v": 0.8, "shear": 3.0, "fliplr": 0.5, "degrees": 45.0}},
        {"name": "Group_50_Augmentation_Set_1_Brightness_6_Shear_0.8_Flip_LR_80_Rotation", "augments": {"hsv_v": 1.0, "shear": 6.0, "fliplr": 0.8, "degrees": 80.0}},
        {"name": "Group_51_Augmentation_Set_0.1_Translate_9_Shear_0.2_Mosaic_110_Rotation", "augments": {"translate": 0.1, "shear": 9.0, "mosaic": 0.2, "degrees": 110.0}}
    ]
    
    run_experiment_batch(my_experiments)