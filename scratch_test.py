import sys
import os

sys.path.insert(0, os.path.abspath('src'))

import chokkhu as ck

dataset_path = 'C:/Users/tamimystic/.gemini/antigravity/brain/5804be46-102d-4f73-a6eb-0fea74ad5b18/scratch/dummy_data'

print("Running image EDA...")
ck.eda.image(dataset_path=dataset_path, save_reports=True, save_dir='scratch/ultra_pro_max_outputs')
print("Image EDA complete.")
