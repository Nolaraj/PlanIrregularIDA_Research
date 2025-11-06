"""It needs to be opened from code editor like Pycharm. It maynot work smoothly with the STKO terminal"""
'''This code is preferred over the Parametric code as it efficiently captures the cmd responses and prevents for the STKO GUI Freezing type of inconvenience'''


import tkinter as tk
from tkinter import filedialog

import os
import subprocess
import time



status_interval = 5
CompletedFile = "DoneAnalysis.txt"

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select a .txt file containng path of all script folder",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    return file_path

def RunCMD(command, interval, index=1):
    folder_path = command[0]
    print(f"Running for {index}. {folder_path}")


    full_command = ' & '.join(command)

    process = subprocess.Popen(
        ['start', 'cmd', '/c', full_command],    #Never use /k instead of /c as it doesnot support the status extraction of the cmd file processing ie end or stopped phase.
        cwd=folder_path,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line.strip())
        process.wait()

        if process.returncode == 0:
            print("Completed Successfully")

            DoneFile = open(AnalysisDonePath, "a")
            DoneFile.write(folder_path + "\n")
            DoneFile.close()

        else:
            print(f"Solver failed for {folder_path}. Return code: {process.returncode}")

    except KeyboardInterrupt:
        print("Monitoring interrupted.")

def PathRefiner(List):
    Paths = []
    for Item in List:
        Item = Item.split("\n")

        path = r""
        for i in Item:
            path = os.path.join(path, i)
        Paths.append(path)
    return Paths

def Analyze():
    # Launcher_File = "LaunchSTKOMonitor.bat"

    Paths_File = open(f'{MainPathFile}', "r")
    Input_Files = Paths_File.readlines()
    Paths = PathRefiner(Input_Files)




    for index, Item in enumerate(Paths):
        DonePaths = []
        if os.path.exists(AnalysisDonePath):
            file1 = open(AnalysisDonePath, "r")
            Input_Files = file1.readlines()
            DonePaths = PathRefiner(Input_Files)
            file1.close()

        if Item not in DonePaths:
            command_to_run = [
                Item ,
                'dir',
                'C:\\OpenSees-Solvers\\openseesmp.bat .\\main.tcl 4'
            ]

            RunCMD(command_to_run, status_interval, index=index + 1)

def getInnermostDirswithSizeLessthanmb(root_dir):
    FileSizeLimit = 2          #5mb
    BranchDeadEnd = "fixed"     #Dead end file, ie the system doesnot iterate after this inside branch and checks only at the end, Write all in Lowercase

    output_file = "files_small5mb.txt"
    output_file = os.path.join(root_dir, output_file)
    max_size = FileSizeLimit * 1024 * 1024  # 5 MB in bytes

    def all_files_under_limit(dirpath, limit):
        """Return True if all files in directory are below limit."""
        for entry in os.scandir(dirpath):
            if entry.is_file():
                if entry.stat().st_size > limit:
                    return False
        return True

    qualified_dirs = []
    unqualified_dirs = []       #It catches the files with the size greater than the limit as specified
    all_dirs =[]

    for current_dir, subdirs, files in os.walk(root_dir):
        # Case 1: If this folder's name is "Fixed"
        if os.path.basename(current_dir).lower() == BranchDeadEnd:
            all_dirs.append(current_dir)
            if all_files_under_limit(current_dir, max_size):
                qualified_dirs.append(current_dir)
            # Don't walk deeper — clear subdirs
            subdirs[:] = []
            continue

        # Case 2: Normal folders — check only if innermost
        if not subdirs:
            if all_files_under_limit(current_dir, max_size):
                qualified_dirs.append(current_dir)

    unqualified_dirs = [x for x in all_dirs if x not in qualified_dirs] #It catches the files with the size greater than the limit as specified
    #qualified_dirs cathes the less than the limit files

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        for d in unqualified_dirs:
            f.write(d + "\n")

    print(f"✅ Done. {len(qualified_dirs)} directories written to {output_file}")

if __name__ == '__main__':
    #Get all the files with size less that 5 mb in the text file
    root_dir = r"E:\Machine Learning Research\Numerical Analysis\Trial 1"
    # getInnermostDirswithSizeLessthanmb(root_dir)

    #Processing for the data to be Analyzed excluding the files that were previosly analyzed
    MainPathFile = open_file_dialog()
    FolderPath = os.path.dirname(MainPathFile)
    AnalysisDonePath = os.path.join(FolderPath, CompletedFile)
    # DonePaths = []
    # if os.path.exists(AnalysisDonePath):
    #     file1 = open(AnalysisDonePath, "r")
    #     Input_Files = file1.readlines()
    #     DonePaths = PathRefiner(Input_Files)
    #     file1.close()

    Analyze()
