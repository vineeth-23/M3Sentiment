import gdown
from pathlib import Path

# url = "https://drive.google.com/file/d/1BuJ21w2BKS5P6Y3sdrUM4OC5AsI8e-xj/view?usp=share_link"
url = "https://drive.google.com/file/d/1_jhuYbVAgVXeTdn1eW8vYtjTyND0LnSO/view?usp=drive_link"
output = Path(__file__).resolve().parents[1] / "aligned_mosei_dataset.pkl"
gdown.download(url, str(output), fuzzy=True)

print(f"File downloaded to {output}")
