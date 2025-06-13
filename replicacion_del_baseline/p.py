from pathlib import Path
import shutil

src_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\nnunet_input_for_prediction_train")
dst_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\nnunet_input_1phase_only")
dst_dir.mkdir(exist_ok=True)

for f in src_dir.glob("*_0000.nii.gz"):
    shutil.copy(f, dst_dir / f.name)

print("✅ Copiados todos los _0000.nii.gz a una nueva carpeta.")
