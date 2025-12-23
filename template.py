import os
from pathlib import Path
import logging

# i don't want to save this log, thats why all log code i set as comments

#folder="log"
#file="demo.log"

#os.makedirs(folder,exist_ok=True)

#path=os.path.join(folder,file)
#print(path)

logging.basicConfig(
    #filename=path,
    format='[%(asctime)s ] %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

while True:
    project_name=input("Enter your project name: ")
    if project_name != "":
        logging.info(f"Created Porject. Your project name is: {project_name}")
        break
    else:
        logging.info("Project name can not be empty. Please try again")

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    #print(filepath)
    filedir,filename=os.path.split(filepath)
    #print(f"folder: {filedir} ======= filename: {filename}")

    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)
        
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for file: {filename}")
        

        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as fp:
                pass
                logging.info(f"Creating a new file: {filename} at path: {filepath}")
        
        else:
            logging.info(f"file is already present at: {filepath}")



