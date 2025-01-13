# Security Camera System

A face recognition-based security system for Linux that monitors and controls access through your computer's camera.

## Installation Guide for Linux

### Prerequisites

1. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
git cmake libopencv-dev
```

### Setting up Python Environment

1. Install pyenv (if not already installed):
```bash
curl https://pyenv.run | bash
```

2. Add pyenv to your shell configuration (~/.bashrc or ~/.zshrc):
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

3. Reload your shell configuration:
```bash
source ~/.bashrc
```

4. Install Python 3.11.0:
```bash
pyenv install 3.11.0
```

5. Create a virtual environment for the project:
```bash
cd to/the/folder/you/want/to/clone/security-script
pyenv local 3.11.0
python -m venv .venv
source .venv/bin/activate
```

### Installing the Project

1. Clone the repository:
```bash
git clone <repository-url>
cd security-script
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Add authorized face images to the `database1` directory (supported formats: .jpg, .jpeg, .png)

2. Run the security camera system:
```bash
python security_linux.py
```

### Usage Notes

- The system will create an `unknown_faces` directory to store detected unauthorized faces
- Logs are stored in `security_system.log`
- Press 'q' to quit the application
- The system will automatically log out the current user if an unauthorized face is detected in consecutive frames

## License

This project is licensed under the MIT License - see the LICENSE file for details.
