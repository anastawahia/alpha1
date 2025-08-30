# Run the Project on RunPod (GPU Pod)

This guide shows how to pull your repo, set up Python, place your data on a **RunPod GPU Pod**, and run the code. No VS Code required.

---

## 0) Create a GPU Pod
1. Log in to RunPod and create a **GPU Pod** (choose a GPU that fits your budget/perf: L4/T4/A40/A100).
2. Pick a **Docker image** with CUDA + Python (e.g., Ubuntu 22.04 base with CUDA 12.x). A common choice is an image that includes NVIDIA drivers and Python preinstalled.
3. **Persistent Volume**: Allocate storage (e.g., 50–200 GB). This keeps your files between restarts.
4. Networking:
   - If you plan to run a web service (e.g., FastAPI/Gradio), enable **HTTP** and set a port (e.g., `7860` or `8000`).
   - SSH access can also be enabled if you prefer using your own terminal client.
5. Launch the pod.

> Persistent data is typically stored under **`/workspace`** (or sometimes **`/runpod-volume`**). Anything outside that may be ephemeral. Always keep your repo and data inside `/workspace`.

---

## 1) Open a Terminal on the Pod
- Use the **Web Terminal** from the RunPod UI (or SSH if enabled).
- Check GPU is visible:
  ```bash
  nvidia-smi
  ```

---

## 2) Prepare directories (persistent)
```bash
# Use the persistent path provided by the image/template; commonly:
cd /workspace
mkdir -p projects
cd projects
```

> If your template mounts a different persistent path (e.g., `/runpod-volume`), use that instead.

---

## 3) Clone your repository
```bash
git clone https://github.com/Anastawahia/alpha1.git
cd alpha1
```

If Git is missing:
```bash
sudo apt update
sudo apt install -y git
```

---

## 4) Python virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `python3-venv` is missing:
```bash
sudo apt install -y python3-venv python3-pip
```

---

## 5) Install dependencies
```bash
pip install -r requirements.txt
```

**System libraries** often needed for PDFs/Images/OCR/ML:
```bash
sudo apt update
sudo apt install -y build-essential libgl1 libglib2.0-0 poppler-utils tesseract-ocr
```

**GPU (PyTorch/CUDA)** — make sure wheel matches your CUDA:
```bash
# Example for CUDA 12.9 (adjust to your pod image CUDA version)
pip install --index-url https://download.pytorch.org/whl/cu129 torch torchvision torchaudio
# Verify:
python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
PY
```

> If CUDA versions differ, install the matching PyTorch wheels from PyTorch’s official instructions.

---

## 6) Data layout (inside the repo)
Keep your data in a **persistent** path. Recommended inside the repo so code can use relative paths:

```
/workspace/projects/alpha1/
├── data/
│   ├── structured/      # Excel, CSV
│   ├── unstructured/    # PDF, TXT, DOCX
│   └── images/          # Images
├── storage/             # (if your app writes FAISS/index/etc.)
└── ...
```

Create directories:
```bash
mkdir -p data/structured data/unstructured data/images storage
```

### Uploading data to the pod
- **RunPod Files** tab: drag-drop into `/workspace/projects/alpha1/data/...`
- Or via `scp` (if SSH enabled) from your local machine:
  ```bash
  scp -r ./local_data user@POD_IP:/workspace/projects/alpha1/data/
  ```

---

## 7) Run the project
Depending on your entry point:

**Python script**
```bash
source .venv/bin/activate
python app.py
```

**Jupyter (optional)**
```bash
pip install notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
# Then forward/allow the port in RunPod and open from your browser.
```

**FastAPI/Gradio/Flask (example)**
```bash
# If using uvicorn (FastAPI)
pip install uvicorn[standard]
uvicorn app:app --host 0.0.0.0 --port 8000
# Expose the same port in the pod settings (e.g., 8000) to access from your browser.
```

Keep it running in the background (optional):
```bash
nohup python app.py > app.log 2>&1 &
# or use tmux/screen
```

---

## 8) Pull latest code later
```bash
cd /workspace/projects/alpha1
git pull
```

If you changed files and want to push them back:
```bash
git add .
git commit -m "Update"
git push origin main
```

---

## 9) Notes on persistence and cleanup
- Anything under `/workspace` (or your pod’s persistent volume) survives restarts and re-creations of the pod.
- Avoid storing large virtualenvs, caches, or model binaries in Git. Use `.gitignore` and keep heavy assets in the data folder or fetch them at runtime.
- If you rebuild/recreate pods, just remount the same persistent volume and your repo/data will still be there.

---

## 10) Troubleshooting
- **Module not found**: ensure venv is active and `pip install -r requirements.txt` completed without errors.
- **CUDA mismatch**: reinstall PyTorch matching the CUDA in the container image.
- **Missing system libraries**: install the packages listed above (PDF, OCR, image libs).
- **Port not reachable**: confirm the port is exposed in the pod settings and your server binds to `0.0.0.0`.
