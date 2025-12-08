# GitHub App Setup Guide

This guide walks you through creating a GitHub App for the OpenCV PR-Agent RAG system.

## Why GitHub App?

While a Personal Access Token (PAT) works for testing, a GitHub App provides:
- Better security (no user credentials)
- Precise webhook filtering
- Clear audit trail
- Production-grade authentication

## Step 1: Create the GitHub App

1. Go to **GitHub Settings** → **Developer settings** → **GitHub Apps**
2. Click **New GitHub App**
3. Fill in:
   - **Name**: `OpenCV PR Agent RAG` (or your preferred name)
   - **Homepage URL**: `https://github.com/HarxSan/opencv`
   - **Webhook URL**: Your ngrok URL + `/webhook` (e.g., `https://abc123.ngrok.io/webhook`)
   - **Webhook secret**: Generate with `openssl rand -hex 32`

## Step 2: Configure Permissions

### Repository Permissions

| Permission | Access Level | Purpose |
|------------|--------------|---------|
| Contents | Read | Read repository files |
| Issues | Read & Write | Post comments |
| Pull requests | Read & Write | Read PR data, post reviews |
| Metadata | Read | Basic repository info |

### Subscribe to Events

Check these events:
- [x] Issue comment
- [x] Pull request

## Step 3: Generate Private Key

1. After creating the app, scroll to **Private keys**
2. Click **Generate a private key**
3. Save the downloaded `.pem` file to `./secrets/github-app-private-key.pem`
4. Set permissions: `chmod 600 ./secrets/github-app-private-key.pem`

## Step 4: Install the App

1. Go to your app's page
2. Click **Install App** in the sidebar
3. Select your organization or account
4. Choose **Only select repositories**
5. Select `HarxSan/opencv` (or your fork)
6. Click **Install**

## Step 5: Configure Environment

Add to your `.env`:

```bash
# GitHub App Configuration
GITHUB_APP_ID=123456  # From app settings page
GITHUB_WEBHOOK_SECRET=your_webhook_secret_here

# Private key is read from file
# Place your .pem file at: ./secrets/github-app-private-key.pem
```

## Step 6: Verify Setup

1. Check webhook deliveries in GitHub App settings
2. Look for successful ping delivery
3. Test with `/review` comment on a PR

## Webhook URL Setup with ngrok

For local development:

```bash
# Install ngrok
# https://ngrok.com/download

# Start tunnel
ngrok http 5000

# Use the HTTPS URL for your webhook
# Example: https://abc123.ngrok.io/webhook
```

For production, use a proper domain with HTTPS.

## Troubleshooting

### Webhook signature validation failed
- Verify `GITHUB_WEBHOOK_SECRET` matches the secret in GitHub App settings
- Check that the webhook is sending SHA-256 signatures

### App not responding to comments
- Check webhook delivery history in GitHub App settings
- Verify the app is installed on the repository
- Check server logs: `docker-compose logs pr-agent-rag`

### Permission denied errors
- Verify the app has required permissions
- Reinstall the app if permissions were changed

## Alternative: Personal Access Token

For quick testing, you can use a PAT instead:

1. Go to **GitHub Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens**
2. Create token with:
   - Repository access: `HarxSan/opencv`
   - Permissions: Contents (Read), Issues (Read/Write), Pull requests (Read/Write)
3. Add to `.env`:
   ```bash
   GITHUB_USER_TOKEN=github_pat_xxxx
   ```

Note: With PAT, you still need to configure webhooks manually in repository settings.