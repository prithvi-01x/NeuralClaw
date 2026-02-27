---
name: github_pr_summary
description: Fetch and summarize open pull requests for any GitHub repository
version: 1.0.0
category: dev
risk_level: LOW
capabilities: ["net:fetch"]
requires_confirmation: false
enabled: true
---

## Instructions

When the user asks to summarize, list, or check pull requests for a GitHub repository:

1. Extract the owner and repo name from the user's request (e.g. "microsoft/vscode")
2. Use `web_fetch` to GET `https://api.github.com/repos/{owner}/{repo}/pulls?state=open&per_page=20`
3. Parse the JSON — each PR has: `number`, `title`, `user.login`, `created_at`, `body`, `draft`, `labels`
4. Format a clean summary:
   - Show total open PR count
   - For each PR: `#number — title (author, X days old)`
   - Mark draft PRs with [DRAFT]
   - If PR body exists, include first sentence only
5. If the API returns 404, tell the user the repo wasn't found or is private
6. If no open PRs, say "No open pull requests"

## Example

User: "show me the open PRs for torvalds/linux"
→ GET https://api.github.com/repos/torvalds/linux/pulls?state=open&per_page=20
→ Format and display summary