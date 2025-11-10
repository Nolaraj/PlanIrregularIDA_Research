from PyMpc import *
from PyMpc.MPC_INTERNALS import run_solver
from opensees.mpc_solver_write_input import write_tcl
import os
import shutil
import subprocess
import time
from pathlib import Path
from subprocess import Popen, PIPE
import signal
import numpy as np

App.clearTerminal()
doc = App.caeDocument()

if not App.hasCurrentSolver():
    raise Exception("No solver defined")
solver_command = App.currentSolverCommand()

# For Analyze Only the MainPathFile should contains folder containg Main.tcl file and Script had been properly written
WriteScriptQ = True
AnalyzeQ = False
status_interval = 5
MainPathFile = "FilePathsSoil"
# Available Names = [S, L1, L2, L3, L4, R]
Building_Name = "S"
ScaleFactors = [ 0.1, 0.2, 0.4, 0.6, 0.8, 1]  # , 1.25, 1.5, 2, 3, 4]
fixedBase = False

Soil_Parameters_Soft = {"Layer1": [1.83, 65352, 196056, 42, 0.35, 0, 100, 0.0],
                        "Layer2": [1.83, 69883, 209649, 42, 0.35, 0, 100, 0.0],
                        "Layer3": [1.83, 133396, 400189, 100, 0.35, 0, 100, 0.0]
                        }

Soil_Parameters_Medium = {"Layer1": [1.79, 55195, 165586, 38, 0.35, 0, 100, 0.0],
                          "Layer2": [1.81, 95811, 287435, 38, 0.35, 0, 100, 0.0],
                          "Layer3": [1.81, 133396, 400189, 100, 0.35, 0, 100, 0.0]
                          }

Soil_Parameters_Hard = {"Layer1": [1.69, 78895, 236686, 17, 0.35, 24, 100, 0.0],
                        "Layer2": [1.97, 152818, 458456, 17, 0.35, 32, 100, 0.0],
                        "Layer3": [1.92, 188263, 564790, 100, 0.35, 32, 100, 0.0]}
#______________________________________________________________________________________________
Soil_Sundhara = {"Layer1": [1.73, 77441, 232326, 19, 0.35, 28, 100, 0.0],
                        "Layer2": [1.82, 121022, 363068, 41, 0.35, 18, 100, 0.0],
                        "Layer3": [1.68, 66964, 200893, 100, 0.35, 15, 100, 0.0]}
Soil_Sankhamul = {"Layer1": [1.63, 78895, 115796, 17, 0.35, 17, 100, 0.0],
                        "Layer2": [1.63, 72795, 218387, 39, 0.35, 20, 100, 0.0],
                        "Layer3": [1.63, 87269, 261807, 100, 0.35, 20, 100, 0.0]}
Soil_Tangal = {"Layer1": [1.87, 101386, 304157, 33, 0.35, 26, 100, 0.0],
                        "Layer2": [1.88, 150406, 451219, 26, 0.35, 22, 100, 0.0],
                        "Layer3": [1.87, 131053, 393159, 100, 0.35, 24, 100, 0.0]}
Soil_Kalikastan = {"Layer1": [1.83, 74226, 222678, 45, 0.35, 35, 100, 0.0],
                        "Layer2": [1.83, 104676, 314027, 52, 0.35, 37, 100, 0.0],
                        "Layer3": [1.84, 118299, 354898, 100, 0.35, 36, 100, 0.0]}
Soil_Khusibu = {"Layer1": [1.63, 30973, 92919, 28, 0.35, 11, 100, 0.0],
                        "Layer2": [1.63, 41897, 125692, 20, 0.35, 18, 100, 0.0],
                        "Layer3": [1.63, 49063, 147188, 100, 0.35, 18, 100, 0.0]}
Soil_Naxal = {"Layer1": [1.88, 45891, 137672, 14, 0.35, 30, 100, 0.0],
                        "Layer2": [1.89, 146808, 440425, 13, 0.35, 23, 100, 0.0],
                        "Layer3": [1.88, 177364, 532091, 100, 0.35, 27, 100, 0.0]}
Soil_Putalisadak= {"Layer1": [1.79, 68174, 204523, 23, 0.35, 5, 100, 0.0],
                        "Layer2": [1.79, 62880, 188641, 28, 0.35, 7, 100, 0.0],
                        "Layer3": [1.80, 52733, 158198, 100, 0.35, 9, 100, 0.0]}
Soil_Gaushala = {"Layer1": [1.59, 57291, 171872, 11, 0.35, 22, 100, 0.0],
                        "Layer2": [1.64, 87638, 262915, 22, 0.35, 16, 100, 0.0],
                        "Layer3": [1.77, 144167, 432501, 100, 0.35, 22, 100, 0.0]}
Soil_Shantinagar = {"Layer1": [1.68, 80646, 241937, 39, 0.35, 29, 100, 0.0],
                        "Layer2": [1.80, 110539, 331618, 35, 0.35, 32, 100, 0.0],
                        "Layer3": [1.77, 93421, 280263, 100, 0.35, 32, 100, 0.0]}
#______________________________________________________________________________________________

Fixed = {"Layer1": [1.69, 78895, 236686, 17, 0.35, 24, 100, 0.0],
         "Layer2": [1.97, 152818, 458456, 17, 0.35, 32, 100, 0.0],
         "Layer3": [1.92, 188263, 564790, 100, 0.35, 32, 100, 0.0]}

if fixedBase:
    Soils = [Fixed]
    Soil_Name = ["Fixed"]
    timeDuration = 50
    numIncr = 5000


else:
    # Soils = [Soil_Parameters_Soft, Soil_Parameters_Medium, Soil_Parameters_Hard]
    # Soil_Name = ["Soft", "Medium", "Hard"]
    Soils = [Soil_Sundhara, Soil_Sankhamul, Soil_Tangal, Soil_Kalikastan, Soil_Khusibu, Soil_Naxal,Soil_Putalisadak, Soil_Gaushala, Soil_Shantinagar ]
    Soil_Name = ["Soil_Sundhara", "Soil_Sankhamul", "Soil_Tangal", "Soil_Kalikastan", "Soil_Khusibu", "Soil_Naxal","Soil_Putalisadak", "Soil_Gaushala", "Soil_Shantinagar" ]
    timeDuration = 50
    numIncr = 5000

Earthquake_Parameters = {"Gorkha": [10, 1900, 9.5]}  # ,
# "Northridge": [12, 1500, 15],
#  "San Fernando" : [11, 3995, 19.975]
# }


Soil1ID = 2
Soil2ID = 3
SOil3ID = 4
SoilIDs = [Soil1ID, Soil2ID, SOil3ID]
UniformExcID = 10
monitorTopID = 11
monitorBottomID = 12
RecorderID = 2
TrainsientAID = 13
GMDefinition = 10

# SoilPara = doc.getPhysicalProperty(Soil1ID)
# SoilPara = doc.getPhysicalProperty(Soil2ID)
# SoilPara = doc.getPhysicalProperty(SOil3ID)

UniformEXc = doc.getAnalysisStep(UniformExcID)
monitorTop = doc.getAnalysisStep(monitorTopID)
monitorBottom = doc.getAnalysisStep(monitorBottomID)
Recorder = doc.getAnalysisStep(RecorderID)
TransientAnalysis = doc.getAnalysisStep(TrainsientAID)
# GM = np.array([22, 23, 3, 3, 43, 4], dtype=float)
EQ_space = doc.getDefinition(GMDefinition)


def ScriptWriter(EarthquakesDict):
    # acc, dt, npts, eqname = load_PEERNGA_record(source_seed)
    fileIndex = 1
    Input_Files = []
    for EqName, EqValues in EarthquakesDict.items():
        for soilIndex, soil in enumerate(Soils):
            SName = Soil_Name[soilIndex]
            for ScaleFactor in ScaleFactors:  # [0:1]:
                i = 0
                for LName, Svalues in soil.items():
                    SoilPara = doc.getPhysicalProperty(SoilIDs[i])

                    # SOil Parameters Udpate
                    SoilPara.XObject.getAttribute("rho").quantityScalar.value = Svalues[0]
                    SoilPara.XObject.getAttribute("refShearModul").quantityScalar.value = Svalues[1]
                    SoilPara.XObject.getAttribute("refBulkModul").quantityScalar.value = Svalues[2]
                    SoilPara.XObject.getAttribute("cohesi").quantityScalar.value = Svalues[3]
                    SoilPara.XObject.getAttribute("peakShearStra").real = Svalues[4]
                    SoilPara.XObject.getAttribute("Optional").boolean = True
                    SoilPara.XObject.getAttribute("frictionAng").real = Svalues[5]
                    SoilPara.XObject.getAttribute("refPress").quantityScalar.value = Svalues[6]
                    SoilPara.XObject.getAttribute("pressDependCoe").real = Svalues[7]
                    SoilPara.commitXObjectChanges()
                    i += 1

                # Earthquake Changes
                UniformEXc.XObject.getAttribute("tsTag").index = GMDefinition
                UniformEXc.commitXObjectChanges()

                # Transient Analysis Step Changes
                TransientAnalysis.XObject.getAttribute("numIncr").integer = numIncr
                TransientAnalysis.XObject.getAttribute("duration/transient").real = timeDuration
                TransientAnalysis.commitXObjectChanges()

                # Eaarthquake data definition
                GM = np.array(EqValues["acc"], dtype=float)
                try:

                    if len(GM) >= 2000:
                        GMs = GM[:2000]
                    else:
                        # Pad with zeros up to 2000 length
                        GMs = np.pad(GM, (0, 2000 - len(GM)), mode='constant', constant_values=0)
                except Exception as e:
                    print(f"Error processing GM: {e}")
                    GMs = np.zeros(2000)
                # print(GM[0:10], EqName)
                for eq_ind, eq in enumerate(GMs):
                    EQ_space.XObject.getAttribute("list_of_values").quantityVector.setValueAt(eq_ind, eq)
                EQ_space.XObject.getAttribute('-factor').boolean = True
                EQ_space.XObject.getAttribute('cFactor').real = ScaleFactor
                EQ_space.XObject.getAttribute('dt').real = EqValues["dt"]

                EQ_space.commitXObjectChanges()

                # Monitor Update
                monitorTop.XObject.getAttribute("Use Custom Name").boolean = True
                monitorTop.XObject.getAttribute(
                    "Custom Name").string = f"{EqName}_{ScaleFactor}_{Building_Name}_{SName}_Top"
                monitorTop.commitXObjectChanges()

                monitorBottom.XObject.getAttribute("Use Custom Name").boolean = True
                monitorBottom.XObject.getAttribute(
                    "Custom Name").string = f"{EqName}_{ScaleFactor}_{Building_Name}_{SName}_Base"
                monitorBottom.commitXObjectChanges()

                # Recorder Update
                Recorder.XObject.getAttribute("name").string = f"{EqName}_{ScaleFactor}_{Building_Name}_{SName}"
                Recorder.commitXObjectChanges()

                # Commissioning of Changes
                doc.dirty = True
                doc.commitChanges()
                App.runCommand("Regenerate", "l")
                App.processEvents()

                # Earthquake, ScaleFactor, BuildingName, BaseCondition

                # Creating Directory
                FolderPath = os.path.join(os.getcwd(), EqName, str(ScaleFactor), Building_Name, SName)
                if os.path.exists(FolderPath) is False:
                    os.makedirs(FolderPath)

                # Writing the Script to the assigned Directory]
                write_tcl(FolderPath)
                Input_Files.append(FolderPath)

                print(f"=============\033[1mFile No {fileIndex} has successfully been written==================\033[0m")
                fileIndex += 1

    # Writing the text file for future
    txtPath = os.path.join(os.getcwd(), f'{MainPathFile}.txt')
    Paths_File = open(txtPath, 'a')
    for path in Input_Files:
        Paths_File.write(path)
        Paths_File.write('\n')


def monitor_process(command, interval, index=1):
    full_command = ' & '.join(command)

    process = subprocess.Popen(
        ['start', 'cmd', '/c', full_command],  # Use the 'start' command to open a new command prompt window
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    try:
        print(f"Running for {index}. {command[0]}")
        while process.poll() is None:
            # Read and print lines from stdout
            for line in process.stdout:
                print(line.strip())

            # Read and print lines from stderr
            for line in process.stderr:
                print(line.strip())

            # Wait for the specified interval
            time.sleep(interval)

        print(process.poll(), "Completed Successfully")
        time.sleep(interval)


    except KeyboardInterrupt:
        # Handle keyboard interrupt (e.g., Ctrl+C)
        print("Monitoring interrupted.")


def Analyze():
    # Launcher_File = "LaunchSTKOMonitor.bat"

    Paths_File = open(f'{MainPathFile}.txt', "r")
    Input_Files = Paths_File.readlines()

    # Running for the process
    # Input_Files = [Input_Files[0]]
    for index, Item in enumerate(Input_Files):
        # Running for Laucnher
        Item = Item.split("\n")
        print(Item)

        path = r""
        for i in Item:
            path = os.path.join(path, i)
        Item = path

        command_to_run = [
            f'cd {Item}',
            'dir',
            'C:\OpenSees-Solvers\openseesmp.bat .\main.tcl 4'
        ]
        monitor_process(command_to_run, status_interval, index=index + 1)


def load_PEERNGA_record(filepath: str):
    """
    Load record in .at2 format (PEER NGA Databases).

    Parameters
    ----------
    filepath : str
        Path to the .at2 file.

    Returns
    -------
    Tuple[np.ndarray, float, int, str]
        - acc (np.ndarray): Acceleration time series (g).
        - dt (float): Time step (s).
        - npts (int): Number of points in the record.
        - eqname (str): Identifier string (Year_Name_Station_Component).

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If the file format is not as expected.
    """
    try:
        with open(filepath, 'r') as fp:
            next(fp)  # Skip header line 1
            line2 = next(fp).strip().split(',')
            if len(line2) < 4:
                raise ValueError("Line 2 format incorrect. Expected Name, Date, Station, Component.")
            date_parts = line2[1].strip().split('/')
            if len(date_parts) < 3:
                raise ValueError("Date format incorrect on Line 2. Expected MM/DD/YYYY.")
            year = date_parts[2]
            eqname = (f"{line2[2].strip()}_comp_{line2[3].strip()}")

            next(fp)  # Skip header line 3
            line4 = next(fp).strip().split(',')
            if len(line4) < 2 or 'NPTS=' not in line4[0] or 'DT=' not in line4[1]:
                raise ValueError("Line 4 format incorrect. Expected NPTS=..., DT=...")
            try:
                npts_str = line4[0].split('=')[1].strip()
                npts = int(npts_str)
                dt_str = line4[1].split('=')[1].split()[0]  # Handle potential extra text
                dt = float(dt_str)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Could not parse NPTS or DT from Line 4: {e}")

            # Read acceleration data efficiently
            acc_flat = [float(p) for line in fp for p in line.split()]
            acc = acc_flat


    except FileNotFoundError:
        raise ValueError("Date format incorrect on Line 2. Expected MM/DD/YYYY.")
    except Exception as e:
        raise ValueError(f"Error parsing file {filepath}: {e}")

    return acc, dt, npts, eqname


if __name__ == '__main__':
    # Check if file exists, else create it
    root_folder = os.getcwd()
    file_path = os.path.join(root_folder, f"{MainPathFile}.txt")
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")  # create an empty file
        print(f"File created: {file_path}")
    else:
        print(f"File already exists: {file_path}")

    # Earthquakes Segregation
    root_folder = r"E:\Machine Learning Research\Numerical Analysis\Earthquakes Materials\Grouped_By_Row_Metadata"
    scaled_files = []  # to store all file paths under each 'Scaled' folder

    # Walk through the entire directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check if the current folder name is 'Scaled'
        if os.path.basename(dirpath).lower() == "scaled":
            for file in filenames:
                full_path = os.path.join(dirpath, file)
                scaled_files.append(full_path)
    # Print or use the collected paths
    print(f"Total Scaled files found: {len(scaled_files)}")
    EarthquakesDict = {}
    for filename in scaled_files:
        acc, dt, npts, eqname = load_PEERNGA_record(filename)
        if eqname not in EarthquakesDict:
            EarthquakesDict[eqname] = {}

        EarthquakesDict[eqname]["acc"] = acc
        EarthquakesDict[eqname]["dt"] = dt
        EarthquakesDict[eqname]["npts"] = npts
        EarthquakesDict[eqname]["eqname"] = eqname
    # print(EarthquakesDict)

    if WriteScriptQ:
        ScriptWriter(EarthquakesDict)
    elif AnalyzeQ:
        Analyze()
    else:
        print("Nothing is performed ")

    print("_________________All Code Executed. End of Process !____________________")
