import os 
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

list_of_files =[
 "src/__init__.py",
 "src/helper.py",
 ".env",
 "requirements.txt",
 "setup.py",
 "app.py",
 "research/trials.ipynb",
    
]

for file in list_of_files:
    file_path = os.path.join(os.getcwd(), file)
    filedir,filename = os.path.split(file_path)
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,'w') as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}")