---
name: upload-github
description: Upload latest LINGQIDATA changes to GitHub. Stages, commits, and pushes all local changes.
---

# Upload LINGQIDATA to GitHub

Use this skill whenever the user asks to upload, push, or sync local data to GitHub for the LINGQIDATA project.

## Project Info

| Key | Value |
|-----|-------|
| Repo paths | `/root/autodl-fs/lingqiData/` or `/root/shared-nvme/lingqiData/` |
| Remote | `git@github.com:CHEN-si-yu/LINGQIDATA.git` (SSH) |
| Branch | `master` |
| Email | `1456964303@qq.com` |
| SSH Key | `/home/claude/.ssh/github_lingqi` |

## Workflow

### 1. Check state
```bash
cd /root/autodl-fs/lingqiData && git status
```

### 2. Review changes
- Check diffs: `git diff` and `git diff --cached`
- Review untracked files — skip anything in .gitignore (Model outputs, training data, parquet/feather files, etc.)
- Verify remote is SSH: `git remote get-url origin` should show `git@github.com:CHEN-si-yu/LINGQIDATA.git`

### 3. Stage
```bash
git add <specific-files>    # prefer explicit paths, avoid git add -A
```

### 4. Commit
Use concise Chinese or English summary. Format:
```
Sync: <what changed>
```

Keep messages under one line when possible.

### 5. Push
```bash
git push origin master
```

## .gitignore reminders
Large/generated dirs excluded: `data/history_*/`, `data/min_adj_*/`, `data/factors/`, `data/targets/`, `Model/*/logs/`, `Model/*/model_*/`, `trainingdata/`, `*.pkl`, `*.pth`, `*.ipynb`

## SSH Setup (if push fails)

This container uses SSH to push (HTTPS port 443 is blocked by firewall). If SSH key is missing:

### 1. Generate key
```bash
ssh-keygen -t ed25519 -C "1456964303@qq.com" -f ~/.ssh/github_lingqi -N ""
```

### 2. Add to GitHub
Copy the public key and add it at https://github.com/settings/keys:
```bash
cat ~/.ssh/github_lingqi.pub
```

### 3. Switch remote to SSH
```bash
git remote set-url origin git@github.com:CHEN-si-yu/LINGQIDATA.git
```

### 4. Configure SSH for root (if using sudo git)
```bash
sudo mkdir -p /root/.ssh
echo 'Host github.com
    HostName github.com
    User git
    IdentityFile /home/claude/.ssh/github_lingqi
    StrictHostKeyChecking no' | sudo tee /root/.ssh/config
sudo chmod 600 /root/.ssh/config
```
