#!/bin/bash

# 🚀 Push ISL Recognition Project to GitHub

echo "=================================="
echo "  PUSH TO GITHUB"
echo "=================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git not initialized!"
    echo "Run: git init"
    exit 1
fi

echo "✅ Git repository initialized"
echo ""

# Check current status
echo "📊 Current status:"
git status --short | head -10
echo ""

# Get GitHub repository URL
echo "Enter your GitHub repository URL:"
echo "Example: https://github.com/username/isl-recognition.git"
read -p "URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ No URL provided!"
    exit 1
fi

echo ""
echo "🔗 Adding remote origin..."
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

echo "✅ Remote added: $REPO_URL"
echo ""

# Push to GitHub
echo "📤 Pushing to GitHub..."
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "  ✅ SUCCESS!"
    echo "=================================="
    echo ""
    echo "Your project is now on GitHub!"
    echo "View it at: $REPO_URL"
    echo ""
else
    echo ""
    echo "=================================="
    echo "  ⚠️  PUSH FAILED"
    echo "=================================="
    echo ""
    echo "Common issues:"
    echo "1. Repository doesn't exist on GitHub"
    echo "   - Create it first at: https://github.com/new"
    echo ""
    echo "2. Authentication failed"
    echo "   - Use personal access token instead of password"
    echo "   - Generate at: https://github.com/settings/tokens"
    echo ""
    echo "3. Repository already has content"
    echo "   - Use: git pull origin main --allow-unrelated-histories"
    echo "   - Then: git push -u origin main"
    echo ""
fi
