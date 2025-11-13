#It is designed solely for copying the .txt files from one root folder to another with preserving its tree structure.
# When the open dialog is opened selected the parent root folder from which the data needs to be extracted and transferred.
#Similarly path of the output folder is where the data are copied back then.

import tkinter as tk
from tkinter import filedialog
import os
import shutil


def open_folder_dialog():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        title="Select the root folder containing the results"
    )

    return folder_path
def list_subfolders(root_folder):
    subfolders = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            subfolder_path = os.path.join(dirpath, dirname)
            subfolders.append(subfolder_path)

    return subfolders
def filePaths(Subfolders, FileName):
    ResultFiles = []
    for subfolder in subfolders:
        for files in os.listdir(subfolder):
            if files.endswith(FileName):
                Filepath = os.path.join(subfolder, files)
                ResultFiles.append(Filepath)

    return ResultFiles



folderpath = open_folder_dialog()
subfolders = list_subfolders(folderpath)
FileName = "Result.txt"
ResultFiles = filePaths(subfolders, FileName)

print(len(ResultFiles))
validfiles = []
invalidfiles = []
for file in ResultFiles:
    if os.path.getsize(file) > 1024:
        validfiles.append(file)
    else:
        invalidfiles.append(file)

Insidefolder = "TS_Outputs"
pathsContainer = {}
for index, file in enumerate(validfiles):
    tspath = os.path.join(os.path.dirname(file), Insidefolder)
    if not os.path.exists(tspath):
        validfiles.remove(file)
        invalidfiles.append(file)
    else:
        pathsContainer[index] = [tspath, file]

# print(pathsContainer)
# print(len(validfiles), len(invalidfiles))

#_________________________________________________________________________________________Output Defination
OutputDir = r"E:\Machine Learning Research\Numerical Analysis\ExtractedResults\New folder"
for key, value  in pathsContainer.items():
    TsFolder = value[0]
    resultfile = value[1]

    newTSpath = TsFolder.replace(folderpath, OutputDir, 1)
    newResultPath  = resultfile.replace(folderpath, OutputDir, 1)
    if not os.path.exists(newTSpath):
        shutil.copytree(TsFolder, newTSpath)

        newResultFolder = os.path.dirname(newTSpath)
        shutil.copy2(resultfile, newResultFolder)

    print(newTSpath)
    break





