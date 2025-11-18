import kagglehub
import os
import shutil

# Download the dataset from kaggle api
path = kagglehub.dataset_download("msambare/fer2013")

# save it to the datasets subdirectory
destination_dir = os.path.join(os.getcwd(), "datasets", "fer2013")
os.makedirs(destination_dir, exist_ok=True)

# for some reason, kaggle saves it to its own directory, so we copy over
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(destination_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
        print("Dataset saved to: ", destination_dir)