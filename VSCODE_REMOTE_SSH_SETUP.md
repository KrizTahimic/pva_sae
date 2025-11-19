# VS Code Remote-SSH Setup Guide for Google Cloud VM

This guide provides detailed step-by-step instructions for setting up Visual Studio Code Remote-SSH to develop and run your PVA-SAE project on a Google Cloud VM.

## Why Use VS Code Remote-SSH?

- **Full IDE Experience**: Get IntelliSense, debugging, and all VS Code features while running code on the VM
- **GPU Access**: Execute code on the VM's GPU without copying files back and forth
- **Seamless Development**: Edit files as if they're local, but they're actually on the remote VM
- **Integrated Terminal**: Terminal commands run directly on the VM
- **Easy Debugging**: Set breakpoints and debug Python code running on remote hardware

---

## Prerequisites

### On Your Local Machine
- ✅ Visual Studio Code installed
- ✅ SSH client installed (comes with macOS/Linux, use Git Bash or WSL on Windows)
- ✅ SSH key pair for Google Cloud authentication

### On Your Google Cloud VM
- ✅ VM is running and accessible
- ✅ SSH access enabled (port 22)
- ✅ Repository already cloned at a known path
- ✅ Conda environment (`pva_sae`) already set up
- ✅ All dependencies installed

---

## Step 1: Get Your VM Connection Details

### 1.1 Find Your VM's External IP ✅

**Your VM External IP:** `34.80.95.244`

~~1. Go to [Google Cloud Console](https://console.cloud.google.com/)~~
~~2. Navigate to **Compute Engine** → **VM Instances**~~
~~3. Find your VM in the list~~
~~4. Copy the **External IP** address~~

### 1.2 Identify Your SSH Username ✅

**Your SSH Username:** `kriz_tahimic`

~~- Your Google Cloud username (visible when you SSH from the browser)~~
~~- Or the username you specified when creating the VM~~
~~- To verify, try connecting from terminal: `gcloud compute ssh <vm-name>`~~

### 1.3 Locate Your SSH Key ✅

**Your SSH Key Paths:**
- **Private key:** `~/.ssh/id_ed25519`
- **Public key:** `~/.ssh/id_ed25519.pub`

~~Google Cloud typically stores SSH keys at:~~
~~- **macOS/Linux**: `~/.ssh/google_compute_engine`~~
~~- **Windows**: `C:\Users\<YourName>\.ssh\google_compute_engine`~~

---

## Step 2: Configure SSH on Your Local Machine ✅

### 2.1 Open SSH Config File ✅

On your **local machine**, open or create the SSH config file:

```bash
# macOS/Linux
nano ~/.ssh/config

# Windows (Git Bash or PowerShell)
notepad ~/.ssh/config
```

### 2.2 Add VM Configuration ✅

Your configuration has been added:

```ssh-config
Host pva-sae-vm
    HostName 34.80.95.244
    User kriz_tahimic
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**Explanation:**
- `Host pva-sae-vm`: Nickname for your VM (you can choose any name)
- `HostName`: The VM's external IP address
- `User`: Your SSH username on the VM
- `IdentityFile`: Path to your SSH private key
- `ForwardAgent`: Allows using your local SSH keys for git on the VM
- `ServerAliveInterval`: Keeps connection alive (prevents timeouts)

### 2.3 Set Correct Permissions (macOS only) ✅

```bash
chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/id_ed25519
```

### 2.4 Test SSH Connection ✅

SSH connection verified successfully! ✅

```bash
ssh pva-sae-vm
```

Connection test output:
```
SSH connection successful!
kriz_tahimic
llama-gpu-instance-b4zb
```

**Troubleshooting:**
- **Permission denied**: Check that your SSH key is correct and has proper permissions
- **Connection refused**: Verify firewall rules allow SSH (port 22)
- **Host not found**: Double-check the IP address
- **Timeout**: Check that the VM is running

---

## Step 3: Install VS Code Remote-SSH Extension ✅

### 3.1 Open VS Code Extensions ✅

~~1. Open Visual Studio Code~~
~~2. Click the **Extensions** icon in the sidebar (or press `Cmd+Shift+X` / `Ctrl+Shift+X`)~~
~~3. Search for: **"Remote - SSH"**~~
~~4. Install the extension by Microsoft (it should be the top result)~~

### 3.2 Verify Installation ✅

Extension installed successfully! You should now see:
- A green/blue icon with `><` symbols in the bottom-left corner of VS Code
- This is the "Remote Connection" indicator

---

## Step 4: Connect to Your VM ✅

### 4.1 Initiate Connection ✅

**Method 1: Using Command Palette (Recommended)**
1. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
2. Type: `Remote-SSH: Connect to Host...`
3. Select it and press Enter
4. Choose `pva-sae-vm` (the host you configured in Step 2)

**Method 2: Using Remote Icon**
1. Click the green `><` icon in the bottom-left corner
2. Select **"Connect to Host..."**
3. Choose `pva-sae-vm`

### 4.2 First-Time Connection

On first connection, VS Code will:
1. Ask you to select the platform: Choose **Linux**
2. Show "Setting up SSH Host..." in the output panel
3. Install VS Code Server on the VM (this takes 1-2 minutes)
4. Reload the window automatically

You may also be prompted:
- **"Are you sure you want to continue?"** → Click **"Continue"**
- **Fingerprint verification** → Click **"Yes"** or type `yes`

### 4.3 Verify Connection

When successfully connected, you should see:
- **Bottom-left corner**: `SSH: pva-sae-vm` (in green)
- **Window title**: Shows the hostname
- **Output panel**: "Running ssh connection command..." followed by success

---

## Step 5: Open Your Project on the Remote VM

### 5.1 Open Folder

1. Click **File** → **Open Folder...**
2. A file browser for the **remote VM** will appear
3. Navigate to where you cloned the repository, typically:
   ```
   /home/your_username/pva_sae
   ```
   Or wherever you ran `git clone`
4. Click **"OK"** or **"Open"**

### 5.2 Trust the Workspace

VS Code will ask: **"Do you trust the authors of the files in this folder?"**
- Click **"Yes, I trust the authors"**

### 5.3 Verify Project Loaded

You should now see:
- The project file tree in the Explorer sidebar
- Files like `CLAUDE.md`, `run.py`, `common/`, `phase*/` directories
- The status bar shows: `SSH: pva-sae-vm`

---

## Step 6: Install Extensions on Remote

**Important:** Extensions need to be installed separately for the remote environment!

### 6.1 Install Python Extension

1. Go to Extensions (`Cmd+Shift+X` / `Ctrl+Shift+X`)
2. Search for: **"Python"**
3. Find "Python" by Microsoft
4. Click **"Install in SSH: pva-sae-vm"** (not "Install" - that's for local)

### 6.2 Install Pylance (Usually Auto-Installed)

Pylance typically installs with the Python extension, but verify:
1. Search for: **"Pylance"**
2. Ensure it shows "Installed in SSH: pva-sae-vm"

### 6.3 Other Recommended Extensions (Optional)

Install these on the remote if you use them:
- **GitLens**: Enhanced git features
- **autoDocstring**: Generate Python docstrings
- **indent-rainbow**: Make indentation visible
- **Error Lens**: Inline error messages

**How to tell if an extension is installed remotely:**
- Extensions show "Installed in SSH: hostname" when remote
- Some extensions only work locally (themes, keybindings)
- Most language/development extensions should be installed remotely

---

## Step 7: Configure Python Interpreter

### 7.1 Select Interpreter

1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Type: `Python: Select Interpreter`
3. Choose **"Enter interpreter path..."**
4. Enter the full path to your conda environment's Python:
   ```
   ~/miniconda3/envs/pva_sae/bin/python
   ```
   Or the full absolute path:
   ```
   /home/your_username/miniconda3/envs/pva_sae/bin/python
   ```

**Alternatively, let VS Code discover it:**
1. Select `Python: Select Interpreter`
2. VS Code might auto-discover conda environments
3. Look for: `Python 3.x.x ('pva_sae': conda)`

### 7.2 Verify Interpreter

Open a Python file (e.g., `run.py`) and check:
- **Bottom-right corner**: Should show `3.x.x ('pva_sae': conda)` or similar
- Click on it to change if incorrect

### 7.3 Test Interpreter

Open integrated terminal and verify:
```bash
which python
# Should show: /home/username/miniconda3/envs/pva_sae/bin/python

python --version
# Should show Python 3.x.x
```

---

## Step 8: Configure Integrated Terminal

The integrated terminal automatically runs on the VM, but you may want to auto-activate the conda environment.

### 8.1 Auto-Activate Conda (Optional but Recommended)

**Option A: Modify Shell RC File on VM**

SSH into the VM and edit your shell configuration:

```bash
# For bash (most common)
nano ~/.bashrc

# For zsh
nano ~/.zshrc
```

Add these lines at the end:

```bash
# Auto-activate pva_sae conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pva_sae
```

Save and exit, then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

**Option B: VS Code Settings (Terminal-Specific)**

1. Open VS Code settings (`Cmd+,` / `Ctrl+,`)
2. Search for: `terminal.integrated.shellArgs`
3. Add custom shell args to source conda

### 8.2 Verify Terminal

1. Open a new integrated terminal in VS Code (`Ctrl+` backtick or Terminal → New Terminal)
2. You should see `(pva_sae)` prefix in the prompt
3. Run:
   ```bash
   which python
   # Should show conda env path

   conda info --envs
   # Should show * next to pva_sae
   ```

---

## Step 9: Verification and Testing

### 9.1 Check GPU Access

In the VS Code integrated terminal:

```bash
nvidia-smi
```

Expected output:
- GPU model (e.g., Tesla T4, A100, etc.)
- Memory information
- CUDA version

**If `nvidia-smi` fails:**
- Verify GPU is attached to the VM in Google Cloud Console
- Check CUDA drivers are installed: `ls /usr/local/cuda*/bin/`

### 9.2 Test Conda Environment

```bash
# Verify environment is activated
conda info --envs

# Test key imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel; print('Transformers OK')"
```

All should succeed without errors.

### 9.3 Run a Test Phase

Test the full pipeline with a small subset:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae && python3 run.py phase 0 --start 0 --end 5
```

This should:
- Run Phase 0 (difficulty analysis)
- Process only the first 5 problems
- Complete without errors

**Expected output:**
```
[Phase 0] Starting difficulty analysis...
[Phase 0] Processing problems 0-5...
[Phase 0] Completed: 5/5 problems
```

### 9.4 Test Debugging

1. Open `run.py`
2. Set a breakpoint (click left of line number, or press F9)
3. Press `F5` or **Run** → **Start Debugging**
4. Select **"Python File"** when prompted
5. The debugger should stop at your breakpoint

---

## Step 10: Best Practices and Tips

### Working with Remote Files

- **All file operations happen on the VM** - you're editing remote files directly
- **Changes are immediate** - no sync delay
- **Use Git normally** - commit/push/pull work as expected
- **Large files** - avoid downloading large data files to local machine

### Managing Connections

**Reconnecting:**
- Click the green `><` icon → "Connect to Host..." → Select your VM
- Or use `Cmd+Shift+P` → "Remote-SSH: Connect to Host..."

**Disconnecting:**
- Click the green `SSH: pva-sae-vm` → "Close Remote Connection"
- Or just close VS Code

**Multiple Sessions:**
- You can open multiple VS Code windows to the same VM
- Each gets its own terminal and file view

### Port Forwarding (For Jupyter, TensorBoard, etc.)

VS Code automatically forwards ports when you run services:

1. Start a service on the VM (e.g., Jupyter on port 8888)
2. VS Code detects it and shows a notification: "Open in Browser"
3. Click to access the service via forwarded port

**Manual port forwarding:**
1. `Cmd+Shift+P` → "Forward a Port"
2. Enter port number (e.g., 8888)
3. Access at `localhost:8888` on your local browser

### File Transfer

**Small files:**
- Use VS Code's file explorer to drag-and-drop

**Large files (datasets, models):**
- Use `scp` or `gsutil` from terminal
- Or download directly on the VM using `wget`/`curl`

### Performance Tips

- **SSH Keep-Alive**: The config from Step 2 prevents disconnections
- **Large Directories**: Opening huge folders can be slow initially
- **Output Channels**: Check "Remote-SSH" output for connection diagnostics
- **Extension Performance**: Only install necessary extensions remotely

---

## Troubleshooting

### Connection Issues

**Problem: "Could not establish connection to pva-sae-vm"**

Solutions:
1. Verify VM is running in Google Cloud Console
2. Test SSH from terminal: `ssh pva-sae-vm`
3. Check `~/.ssh/config` for typos
4. Verify firewall rules allow SSH (port 22)
5. Check VS Code output panel (Remote-SSH) for detailed errors

**Problem: "Permission denied (publickey)"**

Solutions:
1. Verify SSH key path in `~/.ssh/config`
2. Ensure private key has correct permissions: `chmod 600 ~/.ssh/google_compute_engine`
3. Check that public key is in VM's `~/.ssh/authorized_keys`
4. Test with verbose SSH: `ssh -v pva-sae-vm`

**Problem: "Connection timeout"**

Solutions:
1. Verify VM external IP hasn't changed
2. Check VM firewall rules in Google Cloud Console
3. Ensure VPC allows SSH traffic
4. Try from different network (could be local firewall)

### Extension Issues

**Problem: "Python extension not working"**

Solutions:
1. Ensure installed "in SSH: hostname" not just locally
2. Reload VS Code window: `Cmd+Shift+P` → "Reload Window"
3. Check extension host: `Cmd+Shift+P` → "Developer: Show Running Extensions"

**Problem: "Interpreter not found"**

Solutions:
1. Verify conda environment exists: `conda info --envs` in terminal
2. Use absolute path: `/home/username/miniconda3/envs/pva_sae/bin/python`
3. Refresh interpreter list: `Cmd+Shift+P` → "Python: Select Interpreter"

### Terminal Issues

**Problem: "Conda not found in terminal"**

Solutions:
1. Initialize conda: `source ~/miniconda3/etc/profile.d/conda.sh`
2. Add to `~/.bashrc` or `~/.zshrc` (see Step 8.1)
3. Reload terminal: close and open new terminal tab

**Problem: "GPU not accessible"**

Solutions:
1. Verify GPU attached in Google Cloud Console
2. Check NVIDIA drivers: `nvidia-smi`
3. Install CUDA toolkit if missing
4. Verify PyTorch built with CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Performance Issues

**Problem: "VS Code Server slow or unresponsive"**

Solutions:
1. Check VM resources in Google Cloud Console (CPU/RAM usage)
2. Increase VM machine type if underpowered
3. Close unnecessary VS Code windows
4. Disable resource-heavy extensions

**Problem: "File save is slow"**

Solutions:
1. Check network latency: `ping <vm-ip>`
2. Disable auto-save if not needed
3. Use better network connection
4. Check VM disk I/O performance

---

## Quick Reference

### Essential Commands

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae

# Test GPU
nvidia-smi

# Run a phase
python3 run.py phase <N> [OPTIONS]

# Check git status
git status

# Pull latest changes
git pull origin main
```

### VS Code Shortcuts

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| Command Palette | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| Open Terminal | ``Ctrl+` `` | ``Ctrl+` `` |
| Select Interpreter | `Cmd+Shift+P` → Python: Select | `Ctrl+Shift+P` → Python: Select |
| New Terminal | `Ctrl+Shift+` ` | `Ctrl+Shift+` ` |
| Start Debugging | `F5` | `F5` |
| Toggle Breakpoint | `F9` | `F9` |

### Useful VS Code Settings for Remote

Add to `.vscode/settings.json` in your project:

```json
{
  "python.defaultInterpreterPath": "~/miniconda3/envs/pva_sae/bin/python",
  "python.terminal.activateEnvironment": true,
  "files.watcherExclude": {
    "**/data/**": true,
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true
  },
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}"
  }
}
```

This config:
- Sets default Python interpreter
- Auto-activates conda in terminals
- Excludes large directories from file watcher (improves performance)
- Sets PYTHONPATH for imports

---

## Next Steps

Now that you're set up with VS Code Remote-SSH, you can:

1. **Start developing** - Edit files with full IDE support
2. **Run phases** - Execute GPU-intensive tasks directly from VS Code terminal
3. **Debug code** - Set breakpoints and step through execution
4. **Monitor progress** - View logs and outputs in integrated terminal
5. **Commit changes** - Use built-in Git integration

For project-specific commands and workflows, refer to [CLAUDE.md](CLAUDE.md).

---

## Additional Resources

- [VS Code Remote-SSH Documentation](https://code.visualstudio.com/docs/remote/ssh)
- [Google Cloud SSH Documentation](https://cloud.google.com/compute/docs/instances/connecting-to-instance)
- [Conda Environment Management](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

**Happy Coding! 🚀**

If you encounter any issues not covered in this guide, check the troubleshooting section or refer to the official VS Code Remote-SSH documentation.
