# NextGenDA Beginner Installation

## Supported system

The scientifically certified production configuration is:

- Linux x86-64 / AMD64; or
- Windows 10/11 using **WSL2** with Docker Desktop.

Native-Windows production execution is not certified.

## Windows users: open WSL2 first

Do not run `bootstrap_nextgenda.py` from Windows PowerShell.

The exact pinned t-route source contains Linux-valid filenames with `:`
characters. Native Windows NTFS cannot represent those filenames, so the exact
certified source checkout must live in Linux/WSL2.

A Windows Conda environment cannot be reused as the Linux Conda environment
inside WSL2.

## 1. Verify Git inside Linux/WSL2

```bash
git --version
```

## 2. Verify Docker from Linux/WSL2

On Windows, Docker Desktop must have WSL2 integration enabled.

```bash
docker --version
docker info
```

## 3. Install/verify Conda inside Linux/WSL2

```bash
conda --version
```

## 4. Clone NextGenDA in the Linux filesystem

```bash
cd ~
git clone https://github.com/eforoumandi/NextGenDA.git
cd NextGenDA
```

## 5. Create the Linux environment

```bash
conda env create -f environment.yml
conda activate nextgenda
```

## 6. Bootstrap

```bash
python scripts/bootstrap_nextgenda.py
```

The bootstrap validates the Linux/WSL2 host before any external dependency is
cloned or pulled.

## 7. Run

```bash
nextgenda assimilate
```
