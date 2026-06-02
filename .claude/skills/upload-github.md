---
name: upload-github
description: Upload latest LINGQIDATA changes to GitHub. Stages, commits, and pushes all local changes.
---

# Upload LINGQIDATA to GitHub

Use this skill whenever the user asks to upload, push, or sync local data to GitHub for the LINGQIDATA project.

## Project Info

| Key | Value |
|-----|-------|
| Repo path | `/root/shared-nvme/lingqiData/` |
| Remote | `https://github.com/CHEN-si-yu/LINGQIDATA.git` |
| Branch | `master` |
| Email | `1456964303@qq.com` |
| Token | Stored in remote URL (git config) |

## Workflow

### 1. Check state
```bash
cd /root/shared-nvme/lingqiData && git status
```

### 2. Review changes
- Check diffs: `git diff` and `git diff --cached`
- Review untracked files — skip anything in .gitignore (Model outputs, training data, parquet/feather files, etc.)
- Ensure the token is in the remote URL: `git remote get-url origin` should show `https://ghp_...@github.com/CHEN-si-yu/LINGQIDATA.git`

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

If push fails with TLS/GnuTLS errors, retry — it's a transient network issue in this container.

## .gitignore reminders
Large/generated dirs excluded: `data/history_*/`, `data/min_adj_*/`, `data/factors/`, `data/targets/`, `Model/*/logs/`, `Model/*/model_*/`, `trainingdata/`, `*.pkl`, `*.pth`, `*.ipynb`

## Token
The GitHub PAT is embedded in the remote URL so `git push` works without prompting. If the remote URL ever loses the token (e.g., after re-cloning), reset it:
```bash
git remote set-url origin https://<YOUR_GITHUB_PAT>@github.com/CHEN-si-yu/LINGQIDATA.git
```
