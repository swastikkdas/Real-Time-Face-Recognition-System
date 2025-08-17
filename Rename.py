import os

oldname = "Tusky"
newname = "DarkKing"
img_dir = "images"

for filename in os.listdir(img_dir):
    if filename.startswith(oldname + "_"):
        new_filename = filename.replace(oldname + "_", newname + "_", 1)
        os.rename(os.path.join(img_dir, filename), os.path.join(img_dir, new_filename))
