# How to Pull Code and Run in VS Code on Windows

## 1. Install Git
- Download from [https://git-scm.com/download/win](https://git-scm.com/download/win)
- During installation, choose default options.

## 2. Install VS Code
- Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)
- Install and open VS Code.

## 3. Clone the Repository (Pull the Code)
- Open **Command Prompt** or **PowerShell**.
- Navigate to the folder where you want to keep your projects, e.g.:
  ```bash
  cd D:\attarat\projects
  ```
- Run the command:
  ```bash
  git clone https://github.com/Anastawahia/alpha1.git
  ```
- This will create a folder named `alpha1` with the project code.

## 4. Open the Project in VS Code
- In VS Code, go to **File -> Open Folder**.
- Select the folder:
  ```
  D:\attarat\projects\alpha1
  ```

## 5. Create a Virtual Environment (Recommended)
- Open the VS Code terminal (**Ctrl + `**).
- Run:
  ```bash
  python -m venv .venv
  ```
- Activate it:
  ```bash
  .venv\Scripts\activate
  ```

## 6. Install Project Dependencies
- Make sure the virtual environment is active.
- Run:
  ```bash
  pip install -r requirements.txt
  ```

## 7. Run the Project
- Depending on the type of code (Python script, Jupyter Notebook, etc.):
  - **Python script**:  
    ```bash
    python app.py
    ```
  - **Jupyter Notebook**:  
    Open the `.ipynb` file in VS Code and run the cells.

## 8. Data Storage Location
- Create a dedicated folder for your data inside the project, for example:
  ```
  D:\attarat\projects\alpha1\data
  ```
- Suggested structure:
  ```
  data/
  ├── structured/      # Excel, CSV files
  ├── unstructured/    # PDF, TXT, DOCX files
  └── images/          # Image files
  ```


