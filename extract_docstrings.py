import os
import re
import json

def extract_docstrings(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            content = f.read()

    # Regular expression to match class or function name followed by docstring
    pattern = re.compile(r'(def|class)\s+(\w+)\s*\(.*?\):\s*("""(.*?)""")', re.DOTALL)
    matches = pattern.findall(content)

    data = {}
    for match in matches:
        type_, name, _, docstring = match
        docstring = docstring.strip()
        if type_ == 'class':
            data[name] = {"description": docstring, "methods": {}}
            # Extract methods for the class
            method_pattern = re.compile(r'def\s+(\w+)\s*\(self.*?\):\s*("""(.*?)""")', re.DOTALL)
            method_matches = method_pattern.findall(content)
            for m in method_matches:
                m_name, _, m_docstring = m
                data[name]["methods"][m_name] = {"description": m_docstring.strip()}
        else:
            data[name] = {"description": docstring}

    return data

def extract_from_project(root_path, ignore_files=[], ignore_folders=[]):
    project_data = {}

    for root, dirs, files in os.walk(root_path):
        # Skip ignored folders
        dirs[:] = [d for d in dirs if d not in ignore_folders]

        for file in files:
            if file.endswith('.py') and file not in ignore_files:
                file_path = os.path.join(root, file)
                project_data[file] = extract_docstrings(file_path)

    return project_data

if __name__ == "__main__":
    root_path = "C:/Users/Stefan/Desktop/python/Neuro_0"
    ignore_file_list = [".env_example", ".gitignore", "docstrings.json", "make.bat", "Makefile", "readme.md", "requirements.txt"]  # Add filenames to be ignored here
    ignore_folder_list = ["inputs", "venv", ]  # Add folder names to be ignored here

    data = extract_from_project(root_path, ignore_files=ignore_file_list, ignore_folders=ignore_folder_list)

    with open("docstrings.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Extraction complete! Check docstrings.json for the extracted data.")
