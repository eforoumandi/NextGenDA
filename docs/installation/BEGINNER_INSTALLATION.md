# NextGenDA Beginner Installation

This guide starts from a clean computer and ends with a verified NextGenDA
installation ready to launch the public interactive assimilation workflow.

## 1. Certified host contract

The current certified production configuration is:

- Linux `x86_64` / AMD64; or
- Windows 10/11 on an x86-64 machine using **WSL2** with Docker Desktop.

Native Windows, macOS, Linux ARM64, and WSL1 are not certified production
hosts for this release.

The exact pinned t-route source contains Linux-valid filenames with `:`
characters. The certified source checkout must therefore live on a Linux
filesystem.

Do not run `bootstrap_nextgenda.py` from Windows PowerShell.

The bootstrap enforces the Linux/AMD64 host contract before it clones or
pulls external NextGenDA dependencies.

## 2. Windows 10/11: install WSL2

Native Linux AMD64 users should skip to Section 5.

Open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested. Open Ubuntu once and create the Linux account
requested by Ubuntu.

Verify from PowerShell:

```powershell
wsl --status
wsl -l -v
```

The Ubuntu distribution used for NextGenDA must report `VERSION 2`.

## 3. Windows 10/11: install Docker Desktop

Install Docker Desktop for Windows, use the WSL2 backend, and enable WSL
integration for the Ubuntu distribution used for NextGenDA.

Inside Ubuntu/WSL2 verify:

```bash
docker --version
docker info
```

Both commands must succeed.

## 4. Windows/WSL2: install Linux command-line tools

Inside Ubuntu/WSL2:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
```

Verify:

```bash
uname -s
uname -m
git --version
```

Expected certified host identity:

```text
Linux
x86_64
```

Then continue to Section 6.

## 5. Native Linux AMD64: install system tools

The commands below use Ubuntu Linux. On another Linux AMD64 distribution,
install equivalent current packages using the distribution's supported
package manager and Docker's official instructions for that distribution.

Verify the certified host architecture:

```bash
uname -s
uname -m
```

Expected:

```text
Linux
x86_64
```

Install Git and basic HTTPS/download tools:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
```

### Install Docker Engine on Ubuntu

Remove packages that can conflict with Docker's official packages:

```bash
for pkg in   docker.io   docker-doc   docker-compose   docker-compose-v2   podman-docker   containerd   runc
do
  sudo apt-get remove -y "$pkg" 2>/dev/null || true
done
```

Configure Docker's official Ubuntu apt repository:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL   https://download.docker.com/linux/ubuntu/gpg   -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update

sudo apt-get install -y   docker-ce   docker-ce-cli   containerd.io   docker-buildx-plugin   docker-compose-plugin
```

Verify the Docker installation:

```bash
sudo docker run --rm hello-world
```

NextGenDA launches Docker without `sudo`. If your account cannot run
`docker info` without `sudo`, add the current user to the Docker group:

```bash
sudo usermod -aG docker "$USER"
```

Then log out of the Linux session and log back in so the new group membership
takes effect.

Verify:

```bash
docker --version
docker info
```

Both commands must succeed before running the NextGenDA bootstrap.

Docker's official Ubuntu installation and Linux post-installation pages are:

- https://docs.docker.com/engine/install/ubuntu/
- https://docs.docker.com/engine/install/linux-postinstall/


## 6. Install Miniforge inside Linux/WSL2

Do not reuse a native-Windows Conda installation inside WSL2.

Install Linux x86-64 Miniforge:

```bash
cd ~

curl -L \
  -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh \
  -b \
  -p "$HOME/miniforge3"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
conda --version
```

Close and reopen the Linux/WSL2 terminal after `conda init` if needed.

## 7. Clone NextGenDA into the Linux filesystem

For WSL2, keep the repository under the Linux home directory rather than under
`/mnt/c/...`.

```bash
cd ~
git clone https://github.com/eforoumandi/NextGenDA.git
cd NextGenDA
```

Verify:

```bash
git status
git remote -v
```

## 8. Create the certified Python environment

```bash
conda env create -f environment.yml
conda activate nextgenda
```

`environment.yml` installs the NextGenDA package itself. No additional
`pip install` command is required.

Verify:

```bash
python --version
nextgenda --help
```

## 9. Bootstrap external runtime dependencies

Run once after a fresh installation:

```bash
python scripts/bootstrap_nextgenda.py
```

The bootstrap:

1. rejects unsupported hosts before external setup;
2. checks out the exact pinned NGIAB preparation repositories;
3. verifies and pulls the immutable certified runtime container;
4. installs the exact pinned t-route source revision;
5. writes the local runtime configuration; and
6. runs the complete prerequisite checker.

No manual t-route compilation, ngen build, SAC-SMA build, source-path editing,
or runtime-environment `source` command is required for the standard certified
workflow.

## 10. Verify the completed installation

The bootstrap already runs the prerequisite checker. It can be repeated for
diagnostics:

```bash
python scripts/check_prerequisites.py
```

A ready installation should end with:

```text
[PASS] All NextGenDA prerequisites are satisfied.
```

## 11. Start NextGenDA

```bash
nextgenda assimilate
```

## 12. Normal use after installation

Bootstrap is not required before every scientific experiment.

For a normal later session:

```bash
cd ~/NextGenDA
conda activate nextgenda
nextgenda assimilate
```

## 13. Updating an existing installation

```bash
git pull
conda env update -f environment.yml --prune
conda activate nextgenda
python scripts/bootstrap_nextgenda.py
```

After the update/bootstrap completes:

```bash
nextgenda assimilate
```
