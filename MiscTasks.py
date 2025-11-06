import os
import shutil

def CppyingFilestoanotherDirPreservingPathStructure():
    # --- USER INPUTS ---
    path1 = r"E:\Machine Learning Research\Numerical Analysis\Trial 1"
    path2 = r"F:\Trail 1"
    output_txt = r"F:\selected_folders.txt"   # path for saving selected folders

    # --- SETTINGS ---
    endings = ("Soft", "Medium", "Hard")

    selected_folders = []

    # --- STEP 1: Walk through path1 ---
    for root, dirs, files in os.walk(path1):
        for d in dirs:
            if d.endswith(endings):
                full_path = os.path.join(root, d)
                selected_folders.append(full_path)

    # --- STEP 2: Save all selected folder paths into .txt file ---
    with open(output_txt, "w", encoding="utf-8") as f:
        for folder in selected_folders:
            f.write(folder + "\n")

    print(f"✅ Saved {len(selected_folders)} folder paths to {output_txt}")

    # --- STEP 3: Copy those folders preserving structure after 'Trial 1' ---
    for folder in selected_folders:
        # Get relative path after 'Trial 1'
        rel_path = os.path.relpath(folder, path1)
        dest_path = os.path.join(path2, rel_path)

        # Create destination parent dirs
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Copy folder with contents
        if os.path.exists(dest_path):
            print(f"⚠️ Skipping existing folder: {dest_path}")
        else:
            shutil.copytree(folder, dest_path)
            print(f"📁 Copied: {folder} → {dest_path}")

    print("✅ All matching folders copied successfully.")


def getAllfolderpaththatcontainsfilewhichendswith():
    # --- USER INPUT ---
    endswith = ".mpco.cdata"
    path2 = r"F:\Trail 1"
    output_txt = r"F:\folders_with_mpco_cdata.txt"
    
    # --- STEP 1: Collect folders containing .mpco.cdata files ---
    folders_with_cdata = set()
    
    for root, dirs, files in os.walk(path2):
        for file in files:
            if file.endswith(endswith):
                folders_with_cdata.add(root)
                break  # No need to check more files in this folder
    
    # --- STEP 2: Write unique folder paths to .txt file ---
    with open(output_txt, "w", encoding="utf-8") as f:
        for folder in sorted(folders_with_cdata):
            f.write(folder + "\n")
    
    print(f"✅ Found {len(folders_with_cdata)} unique folders containing '.mpco.cdata' files.")
    print(f"📝 Folder paths saved to: {output_txt}")

CppyingFilestoanotherDirPreservingPathStructure()
getAllfolderpaththatcontainsfilewhichendswith()

