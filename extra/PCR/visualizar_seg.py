import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_patient(patient_id, cropped_data_dir):
    # Directorios
    img_dir = Path(cropped_data_dir) / "images" / "test_split" / patient_id
    seg_dir = Path(cropped_data_dir) / "segmentations" / "test_split" / patient_id

    # Cargar first post-contrast (0001)
    post_img_file = img_dir / f"{patient_id.lower()}_0001_cropped.nii.gz"
    if not post_img_file.exists():
        print(f"❌ No post-contrast image found for {patient_id}")
        return
    
    post_img = nib.load(post_img_file).get_fdata()

    # Cargar segmentación
    seg_files = list(seg_dir.glob("*_seg_cropped.nii.gz"))
    if len(seg_files) == 0:
        print(f"❌ No segmentation found for {patient_id}")
        return
    mask = nib.load(seg_files[0]).get_fdata()

    # Slice central
    central_slice = post_img.shape[2] // 2

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(post_img[:, :, central_slice], cmap='gray')
    ax.imshow(mask[:, :, central_slice], cmap='Reds', alpha=0.4)
    ax.set_title(f'{patient_id} - Post-contrast + Mask')
    ax.axis('off')
    plt.show()

# USO:
cropped_data_dir = r"C:\Users\usuario\Documents\Mama_Mia\cropped_data"
visualize_patient("DUKE_019", cropped_data_dir)
