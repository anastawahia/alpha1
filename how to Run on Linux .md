# Run the Project on Linux 

This guide shows how to pull the code from GitHub and run it on Linux using terminal only.

## 1) Install prerequisites
Make sure you have Git and Python 3.10+ (with `venv`) installed.

**Debian/Ubuntu:**
```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

**Fedora:**
```bash
sudo dnf install -y git python3 python3-venv python3-pip
```

**Arch:**
```bash
sudo pacman -Syu --noconfirm git python python-pip
# venv is in the stdlib; no extra package usually needed
```

## 2) Choose a projects folder
Pick a directory to store your projects, e.g. `~/projects`:
```bash
mkdir -p ~/projects
cd ~/projects
```

## 3) Clone the repository
```bash
git clone https://github.com/Anastawahia/alpha1.git
cd alpha1
```

## 4) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> You should see `(.venv)` at the start of your shell prompt.  
> To deactivate later: `deactivate`

## 5) Upgrade pip and install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6) Prepare data directories
Recommended structure inside the repo:
```
data/
├── structured/      # Excel, CSV files
├── unstructured/    # PDF, TXT, DOCX files
└── images/          # Image files
```
Create them:
```bash
mkdir -p data/structured data/unstructured data/images
```

## 7) Run the project
Depends on your entry point:

**Python script (e.g., `app.py`)**
```bash
python app.py
```

**Jupyter Notebook (optional)**
```bash
pip install notebook
jupyter notebook
# Then open the .ipynb file from the browser and run cells
```

## 8) Pull latest changes later
When you want the newest code:
```bash
git pull
```

## 9) Common issues
- **Permissions**: If a script needs execute permission:
  ```bash
  chmod +x script.sh
  ```
- **Missing system libs** (for image/PDF/ML processing): you may need extras such as:
  ```bash
  # Debian/Ubuntu examples
  sudo apt install -y build-essential libgl1 libglib2.0-0 poppler-utils tesseract-ocr
  ```
- **GPU/CUDA**: If you plan to use GPU, install the correct CUDA drivers/toolkit for your distro and make sure your Python packages match the CUDA version.

---

**Data location reminder**  
Keep your dataset inside the repo at `./data`. If you prefer a different path (e.g., `/data/alpha1`), update your code/config accordingly.
