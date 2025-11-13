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

def Copier():
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
    OutputDir = r"E:\Machine Learning Research\Numerical Analysis\ExtractedResults\From PremBhai\Extracted Results"
    for key, value in pathsContainer.items():
        TsFolder = value[0]
        resultfile = value[1]

        newTSpath = TsFolder.replace(folderpath, OutputDir, 1)
        newResultPath = resultfile.replace(folderpath, OutputDir, 1)
        if not os.path.exists(newTSpath):
            shutil.copytree(TsFolder, newTSpath)

            newResultFolder = os.path.dirname(newTSpath)
            shutil.copy2(resultfile, newResultFolder)

        print(newTSpath)

def findMissingResults():
    Buildings = ["S", "L1", "L2", "L3", "L4", "R"]
    ScaleFactor  = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    Soils = ["Fixed", "Soft", "Medium", "Hard"]
    folderpath = open_folder_dialog()
    pathUptoEq = []
    for earthquakes in os.listdir(folderpath):
        full_path = os.path.join(folderpath, earthquakes)
        if os.path.isdir(full_path):  # ✅ only add if it's a folder
            pathUptoEq.append(full_path)
    
    
    missingDatapaths = []
    for eqpath in pathUptoEq:
        for sf in ScaleFactor:
            for building in Buildings:
                for soil in Soils:
                    lastdir = os.path.join(eqpath, str(sf), building, soil)
                    if not os.path.exists(lastdir):
                        print(lastdir)
                        missingDatapaths.append(lastdir)
                    # print(lastdir)

    print(len(missingDatapaths))

# Copier()
# findMissingResults





