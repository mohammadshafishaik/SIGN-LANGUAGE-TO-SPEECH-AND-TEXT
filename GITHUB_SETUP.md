# 🚀 GitHub Setup Guide

## Step 1: Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `isl-recognition` (or your choice)
3. Description: "Real-time Indian Sign Language Recognition with Speech Output"
4. Choose: **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license
6. Click **"Create repository"**

## Step 2: Push Your Code

### Option A: Using the Script (Easiest)

```bash
./push_to_github.sh
```

Enter your repository URL when prompted:
```
https://github.com/YOUR_USERNAME/isl-recognition.git
```

### Option B: Manual Commands

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/isl-recognition.git

# Push to GitHub
git push -u origin main
```

## Step 3: Authentication

GitHub requires authentication. You have two options:

### Option 1: Personal Access Token (Recommended)

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Name: "ISL Recognition"
4. Select scopes: **repo** (all)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

When pushing, use:
- Username: Your GitHub username
- Password: **Paste the token** (not your GitHub password)

### Option 2: SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys
```

Then use SSH URL:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/isl-recognition.git
git push -u origin main
```

## Step 4: Verify

Visit your repository:
```
https://github.com/YOUR_USERNAME/isl-recognition
```

You should see all your files!

## 📝 Next Steps

After pushing to GitHub:

1. **Add README badges** (optional)
2. **Add LICENSE file**
3. **Enable GitHub Pages** for documentation
4. **Add collaborators** if working in a team

## 🔄 Future Updates

After making changes:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push
```

## ⚠️ Troubleshooting

### "Repository not found"
- Make sure you created the repository on GitHub first
- Check the URL is correct

### "Authentication failed"
- Use personal access token, not password
- Generate token at: https://github.com/settings/tokens

### "Updates were rejected"
- Someone else pushed changes
- Run: `git pull origin main`
- Then: `git push`

### "Large files"
- Model files are in .gitignore (they're too large)
- Use Git LFS for large files: https://git-lfs.github.com/

---

**Ready to push? Run: `./push_to_github.sh`** 🚀
