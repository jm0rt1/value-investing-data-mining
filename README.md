

## Summary:
This project was developed on M1 (Arm64 based) based Mac, using Microsoft's Visual Studio Code Version: 1.72.0. The intention is to be compatible with any system through Python's multiplatform nature. However, this project will only be tested on Windows 10 amd64 machine and the latest MacOS version on an Arm64 machine. 

## Scope:
    1. Definitions
    2. Prerequisites
    3. Initializing the Virtual Environment
    4. Running The Program From a Shell
    5. Running The Program
    6. Testing The Code base
    7. UML
## **1. Definitions:**
1) `"./" or "this directory"`- meaning the top level directory of this project, which happens to be the directory that this README.md file has been saved. The README.md is at the top level of this Python project, the project is in a folder named "James_Mortensen_Assignment_1"
2) `"VSCODE"` - Visual Studio Code

## **2. Prerequisites**
The *.sh scripts in this repository should be runnable on MacOS and Windows, as tested via the following shell programs.

### Shell Programs:
These shell programs have been integrated with VSCODE using its Integrated Terminal features:
- Windows: MinGW
- MacOS: Zsh

Set one of these programs as the integrated terminal in VSCODE

**NOTE:** It is not in the socpe of this document to show how to connect these terminals with VSCODE, since this is straightforward and documented online with plenty of support.



### Development Enviornment and Interpreter:
- Visual Studio Code + Extenstions: "Pylance v2022.10.30" and "Python v2022.16.1"
- Python 3.7.3: https://www.python.org/downloads/release/python-373/


## **4. Initializing The Virtual Environment**

### Procedure:
In Visual Studio Code, using a system level install of Python 3.7.3.

1) Give executable rights to the script "./init-venv.sh".
2) In MinGW or Zsh run:
    ```
    ./init-venv.sh
    ```
3) A directory titled "./venv" should now be created.
4) At this time the python extension in VSCODE should show an available interpreter called "venv" located at ./venv.

4) "./.vscode/settings.json" includes a setting that should automatically activate the selected virtual environment when a new integrated terminal is created (use Ctrl + ` to invoke a new terminal). 

    a) if VSCODE does not automatically activate the virtual environment, then it may be necessary to run the following command manually:
    ```
    source ./venv/bin/activate
    ```

"(venv)" should now show at the beginning of each terminal line.

##  **5. Running The Program**

Bundled with the source code, as stated before, is the ./.vscode folder. This folder contains a "launch.json" file. This file is used to configure the Python extension to debug the program from the program entry point ./run.py.

./run.py is used to tell the interpreter that the top level of the project is the folder ".".

run.py can be invoked one of two ways, either through VSCODE's integrated debug features, or via the virtual environment's interpreter via the shell.

### Shell and Python Interpreter Run Procedure:
1) Follow the procedure to initialize and activate the virtual environment.
2) Run the following command:
    ```
    python run.py
    ```

### VSCODE Debugging
1) Change to the debugging sidebar view.
2) Next to the green arrow at the top of the sidebar, ensure "run.py" is selected as the current configuration.
3) Place a breakpoint in the program
4) Click the green arrow 
5) Program should be running in debug mode at this time.


### Running the Executables

Future projects may provide a "frozen" version of the program, providing an executable package to the end user, which will bundle the interpreter with the executable as one set.

This is a placeholder for the procedure.



## **6. Testing Code Base**

Follow the above procedures for initializing the virtual environment, then run:

    ./test.sh

## **7. UML**
The UML Diagrams for this program have been provided in the folder `./docs/UML`.

The model was developed using StarUML Educational License. 2 output images were produced in .png and .jpg formats.

Models.mdj is the raw file that can be edited using StarUML.
