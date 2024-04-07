Project Overview

This project consists of two main Python files, UIC_Models.py and App.py, which should be used in sequence for the intended functionality. 
UIC_Models.py is used first to set up models or perform initial computations, and App.py serves as the application's main entry point, utilizing the setups or models defined previously.

File Descriptions
UIC_Models.py: Contains the definitions for models and possibly initial data loading routines. It references the following paths that you may need to adjust:
CSV and JSON data files (e.g., /UIC_baseball.csv, /batter_name_to_id_mapping.json). These paths are used for data loading and should be set to the correct location of your data files.

A generic /Code directory path, which might be used for referencing code modules or additional scripts.
App.py: The main application file that likely imports and utilizes functions or classes from UIC_Models.py. It contains a reference to:
A /Code directory, which could be used similarly to UIC_Models.py for code organization or module referencing.

Setup Instructions
Environment Setup: Ensure you have a Python environment ready with all necessary packages installed. The required packages might be listed in a requirements.txt file if provided.

File Placement: Place UIC_Models.py and App.py in your project directory. Ensure that the directory structure matches any expected paths coded into these files.

Adjusting Paths: Based on the initial analysis, you will need to update the paths within the files to match your project directory structure. Specifically, look for the following:
Data file paths in UIC_Models.py for CSV and JSON files. Replace /UIC_baseball.csv and /batter_name_to_id_mapping.json with the correct paths to your data files.
The /Code directory reference in both files may need to be updated to reflect where your code modules or scripts are located.
Execution Order: Run UIC_Models.py first to ensure all models and data are correctly set up. Afterward, App.py can be executed to start your application.
