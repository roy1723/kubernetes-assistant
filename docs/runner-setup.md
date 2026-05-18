# CI/CD Setup Guide

This document explains the one-time setup needed for the CI/CD pipeline to work.

## 1. Make the GitHub repository

If you haven't yet:

1. Go to https://github.com/new
2. Repository name: `kubernetes-assistant`
3. Owner: `roy1723`
4. Set to **Public** (or Private with evaluator access)
5. Don't initialize with a README — you have one to push
6. Click "Create repository"

## 2. Create a Personal Access Token (PAT) for GHCR

GHCR is GitHub's Docker registry. Pushing to it requires a token with `write:packages` scope.

1. Go to https://github.com/settings/tokens?type=beta
2. Click **Generate new token** -> **Fine-grained personal access token**
3. **Token name**: `ghcr-kubernetes-assistant`
4. **Resource owner**: roy1723
5. **Expiration**: 90 days (or whatever you prefer)
6. **Repository access**: Only select repositories -> pick `kubernetes-assistant`
7. **Permissions** -> **Repository permissions**:
   - Contents: Read
   - Metadata: Read
   - Packages: **Read and write**
8. Click **Generate token**
9. **Copy the token immediately**. You won't see it again.

## 3. Add the token as a repository secret

1. Go to https://github.com/roy1723/kubernetes-assistant/settings/secrets/actions
2. Click **New repository secret**
3. **Name**: `GHCR_TOKEN`
4. **Secret**: paste the token from step 2
5. Click **Add secret**

The workflow now has access to this secret via `${{ secrets.GHCR_TOKEN }}`. The
token never appears in logs.

## 4. Register your laptop as a self-hosted runner

The runner is a small program from GitHub that runs on your machine, waiting
for jobs. When the workflow needs to run on `self-hosted`, GitHub sends the job
to your runner.

1. Go to https://github.com/roy1723/kubernetes-assistant/settings/actions/runners
2. Click **New self-hosted runner**
3. **Runner image**: Windows
4. **Architecture**: x64

GitHub will show you a sequence of commands. Run them in **PowerShell as
Administrator** (NOT cmd):

```powershell
# Create a folder
mkdir C:\actions-runner; cd C:\actions-runner

# Download (URL will be shown on the page; copy from there - version changes)
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/vX.Y.Z/actions-runner-win-x64-X.Y.Z.zip -OutFile actions-runner.zip

# Extract
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner.zip", "$PWD")

# Configure (the token will be shown on the GitHub page)
./config.cmd --url https://github.com/roy1723/kubernetes-assistant --token <REGISTRATION_TOKEN>
```

During `config.cmd`, when asked:
- **Runner name**: press Enter (default = your machine name)
- **Labels**: type `kubernetes-assistant` and press Enter
  This label is what the workflow uses to target this specific runner.
- **Work folder**: press Enter (default `_work`)

Then run the runner. Two options:

**Option A: As a one-off process (simpler for testing)**

```powershell
cd C:\actions-runner
./run.cmd
```

Leave this window open. The runner is now listening for jobs.

**Option B: Install as a Windows service (more durable)**

```powershell
cd C:\actions-runner
./svc.sh install
./svc.sh start
```

The service starts automatically on boot.

## 5. Verify runner is online

Go back to https://github.com/roy1723/kubernetes-assistant/settings/actions/runners

You should see your runner listed with status **Idle** (green dot).

If it's offline (red dot), the runner process isn't running. Check the
`run.cmd` window or service status.

## 6. Pre-requisites on the runner machine

The CI workflow runs Ollama-dependent steps on this runner. Before triggering
CI, make sure:

- [ ] Ollama is installed and running on the host (`ollama list` works)
- [ ] Both `phi3:mini` and `phi3-kubernetes` are registered (`ollama list` shows them)
- [ ] Docker Desktop is running
- [ ] Python 3.11 is available on PATH
- [ ] The project repo is checked out somewhere accessible

The runner's working directory will be `C:\actions-runner\_work\kubernetes-assistant\kubernetes-assistant`.
This is separate from your dev checkout — that's fine.

## 7. Trigger the first run

After everything above is in place:

```bash
git add .
git commit -m "Add CI/CD pipeline"
git push origin main
```

Go to https://github.com/roy1723/kubernetes-assistant/actions to watch the run.

## 8. Expected first-run problems

This is normal. Common ones:

| Problem | Fix |
|---|---|
| `ruff` reports a long list of errors | Run `ruff check . --fix` locally, commit the fixes |
| `mypy` reports type errors | Add type hints or `# type: ignore` to specific lines |
| Evaluate stage fails: Ollama not reachable | Start Ollama on the runner machine |
| Docker build fails: layer push denied | GHCR_TOKEN doesn't have `write:packages`; recreate it |
| Deploy stage fails: port already in use | Local Docker stack is running; stop it before CI deploys |

## 9. Adding the status badge to README

After the first successful run, add this to the top of README.md:

```markdown
![CI/CD](https://github.com/roy1723/kubernetes-assistant/actions/workflows/ci.yml/badge.svg)
```

This renders as a live badge that turns green/red based on the latest workflow
status on `main`.
