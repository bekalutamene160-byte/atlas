
# Deployment Guide — ATLAS on Render

This guide walks you through deploying ATLAS to Render's free tier. No Docker installation required on your laptop — Render builds and runs the container on their servers.

## Prerequisites

1. A GitHub account with your ATLAS repo pushed
2. A [Render account](https://render.com) (free tier is fine)
3. Your API keys ready:
   - `GROQ_API_KEY` from [console.groq.com](https://console.groq.com)
   - `JINA_API_KEY` from [jina.ai](https://jina.ai)

## Step-by-Step Deployment

### Step 1: Ensure your repo is ready

Verify these files exist in your repo root:
```
Dockerfile
render.yaml
requirements.txt
.env.example
src/atlas/
data/vector_store.db
```

If any are missing, go back and create them.

### Step 2: Commit and push to GitHub

```bash
git add .
git commit -m "Day 7: production deployment config (Dockerfile, render.yaml, README)"
git push origin main
```

### Step 3: Create a Render web service

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub account if you haven't
4. Select your `atlas` repository
5. Render reads `render.yaml` and pre-fills the config

### Step 4: Configure environment variables

In the Render dashboard, scroll to **Environment** and add:

| Key | Value | Notes |
|-----|-------|-------|
| `GROQ_API_KEY` | `gsk_your_real_key` | Secret — never commit this |
| `JINA_API_KEY` | `jln_your_real_key` | Secret — never commit this |

Leave the other env vars as defaults from `render.yaml`.

### Step 5: Deploy

1. Click **Create Web Service**
2. Wait 3-5 minutes for the build to complete
3. You'll see logs like:
   ```
   ==> Building from Dockerfile...
   ==> Installing dependencies...
   ==> Deploying...
   ==> Your service is live at https://atlas-rag-agent.onrender.com
   ```

### Step 6: Verify deployment

```bash
# Health check
curl https://atlas-rag-agent.onrender.com/health
# Expected: {"status":"ok","uptime_seconds":...,"agent_ready":true}

# API info
curl https://atlas-rag-agent.onrender.com/
# Expected: {"name":"ATLAS RAG Agent","version":"1.0.0",...}

# Test chat
curl -X POST https://atlas-rag-agent.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the maximum operating pressure of the Atlas-3 pump?"}'
```

### Step 7: Check logs if something fails

In the Render dashboard:
1. Click your service name
2. Go to **Logs** tab
3. Look for errors in real-time

Common issues:
- **`GROQ_API_KEY not set`** → you forgot Step 4
- **`ModuleNotFoundError`** → check `requirements.txt` has all dependencies
- **`FileNotFoundError: data/vector_store.db`** → the `data/` folder isn't committed

## Post-Deployment

### Custom domain (optional)

1. In Render dashboard → Settings → Custom Domains
2. Add your domain (e.g., `atlas.yourdomain.com`)
3. Add the CNAME record Render gives you to your DNS provider

### Upgrade to paid tier (when ready)

The free tier sleeps after 15 minutes of inactivity. For a portfolio demo, this is fine. For production:
- **Starter ($7/mo)**: always-on, no sleep
- **Standard ($25/mo)**: 2GB RAM, faster CPU

### Monitor usage

- Render dashboard shows CPU/RAM/recent requests
- Groq console shows token usage: [console.groq.com/settings/limits](https://console.groq.com/settings/limits)
- Set up alerts in Render for 5xx errors

## Troubleshooting

### Build fails

Check the build logs in Render. Common causes:
- Missing dependency in `requirements.txt`
- Syntax error in Python code
- Dockerfile syntax error

### App starts but crashes immediately

Check the runtime logs. Common causes:
- Missing env var (`GROQ_API_KEY`, `JINA_API_KEY`)
- Vector store file missing from Docker image
- Port mismatch (Render provides `$PORT`, your app must listen on it)

### Health check fails

- Verify `/health` endpoint returns 200
- Check that the agent initializes successfully on startup
- Look at startup logs for errors

### 502 Bad Gateway

- Your app crashed — check logs
- The first request after sleep takes 30+ seconds — upgrade to paid tier or send a warmup request

## Rollback

Render keeps your last 5 deploys. To rollback:
1. Dashboard → your service → **Deploys** tab
2. Find the last working deploy
3. Click **Rollback to this deploy**
````
