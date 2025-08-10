# 🚂 Railway Deployment Guide for MCP Meme Bot

Deploy your MCP Meme Bot to Railway for reliable, scalable hosting with automatic HTTPS and custom domains.

## 🚀 Quick Railway Deployment

### Option 1: One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/8sY-5r)

### Option 2: Manual Deployment

1. **Create Railway Account**

   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy from GitHub**

   ```bash
   # Push your code to GitHub first
   git add .
   git commit -m "Railway deployment setup"
   git push origin main
   ```

3. **Create New Project on Railway**

   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect the Dockerfile

4. **Set Environment Variables**

   - Go to your project → Variables
   - Add these required variables:

   ```env
   AUTH_TOKEN=your_secure_token_here
   MY_NUMBER=your_phone_number
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. **Deploy**
   - Railway automatically builds and deploys
   - Get your public URL from the project dashboard

## 🔧 Environment Variables

### Required Variables

| Variable         | Description                          | Example                    |
| ---------------- | ------------------------------------ | -------------------------- |
| `AUTH_TOKEN`     | Secure authentication token          | `your_secure_random_token` |
| `MY_NUMBER`      | Your phone number (required by Puch) | `+1234567890`              |
| `GEMINI_API_KEY` | Google Gemini API key                | `AIza...`                  |

### Optional Variables

| Variable   | Description                               | Default   |
| ---------- | ----------------------------------------- | --------- |
| `MCP_HOST` | Server host                               | `0.0.0.0` |
| `MCP_PORT` | Server port (Railway overrides with PORT) | `8086`    |

### Railway Auto-Generated Variables

Railway automatically provides:

- `PORT` - The port your app should listen on
- `RAILWAY_ENVIRONMENT` - Deployment environment
- `RAILWAY_PUBLIC_DOMAIN` - Your app's public URL

## 📋 Railway Configuration

The `railway.toml` file configures Railway deployment:

```toml
[build]
builder = "dockerfile"
buildContext = "."

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 10

[environment]
PORT = "8086"
```

## 🔗 Accessing Your MCP Server

After deployment, your MCP server will be available at:

```
https://your-app-name.railway.app/mcp/
```

### MCP Connection Command

```bash
/mcp connect https://your-app-name.railway.app/mcp Bearer your_auth_token
```

## 🔍 Monitoring and Debugging

### View Logs

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and select project
railway login
railway environment

# View live logs
railway logs --follow
```

### Health Check

Railway automatically monitors `/health` endpoint:

```bash
curl https://your-app-name.railway.app/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "MCP Meme Bot",
  "version": "1.0.0",
  "timestamp": 1699123456.789
}
```

## 🛠 Development Workflow

### Local Development

```bash
# Set environment variables
export AUTH_TOKEN=your_token
export MY_NUMBER=your_number
export GEMINI_API_KEY=your_key

# Run locally
python meme-bot-mcp/mcp-bearer-token/mcp_starter.py
```

### Deploy Changes

```bash
# Commit and push changes
git add .
git commit -m "Update features"
git push origin main

# Railway auto-deploys on push
```

## 🔧 Custom Domain (Optional)

1. **Go to Project Settings**

   - Select your project on Railway
   - Go to Settings → Domains

2. **Add Custom Domain**

   - Click "Add Domain"
   - Enter your domain (e.g., `meme-bot.yourdomain.com`)

3. **Configure DNS**

   - Add CNAME record: `meme-bot` → `your-app.railway.app`
   - Railway provides automatic HTTPS

4. **Update MCP Connection**
   ```bash
   /mcp connect https://meme-bot.yourdomain.com/mcp Bearer your_auth_token
   ```

## 📊 Scaling and Performance

### Railway Plans

- **Hobby Plan**: $5/month
  - 8GB RAM, 8 vCPU
  - Perfect for MCP bots
- **Pro Plan**: $20/month
  - 32GB RAM, 32 vCPU
  - For high-traffic usage

### Auto-Scaling

Railway automatically handles:

- Load balancing
- Auto-restarts on failure
- Health check monitoring
- Zero-downtime deployments

## 🔒 Security Features

### Built-in Security

- **HTTPS by default** - All Railway apps get automatic HTTPS
- **Environment isolation** - Secure environment variable management
- **DDoS protection** - Built-in DDoS mitigation
- **Access logs** - Comprehensive request logging

### Additional Security

Our enhanced middleware provides:

- **Rate limiting** - 100 requests/minute per IP
- **Token validation** - Secure bearer token authentication
- **Security headers** - XSS protection, content type sniffing prevention
- **Input validation** - All inputs are sanitized

## 🚨 Troubleshooting

### Common Issues

1. **Build Fails**

   ```bash
   # Check build logs
   railway logs --build

   # Verify Dockerfile syntax
   docker build -t test .
   ```

2. **Health Check Fails**

   ```bash
   # Test health endpoint
   curl https://your-app.railway.app/health

   # Check if server is listening on correct port
   railway logs | grep "Starting MCP server"
   ```

3. **Authentication Issues**

   ```bash
   # Verify environment variables
   railway variables

   # Test with curl
   curl -H "Authorization: Bearer your_token" \
        https://your-app.railway.app/mcp/
   ```

4. **Memory Issues**
   - Upgrade to Pro plan for more RAM
   - Check logs for memory usage
   - Optimize image processing

### Support

- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Railway Support**: help@railway.app

## 🎯 Production Checklist

- [ ] Set secure `AUTH_TOKEN` (use password generator)
- [ ] Configure `MY_NUMBER` correctly
- [ ] Add valid `GEMINI_API_KEY`
- [ ] Test health endpoint
- [ ] Verify MCP connection
- [ ] Set up custom domain (optional)
- [ ] Enable monitoring alerts
- [ ] Test all meme tools functionality
- [ ] Configure backup/recovery plan

## 📈 Usage Monitoring

### Railway Analytics

Railway provides built-in metrics:

- Request volume
- Response times
- Error rates
- Memory/CPU usage

### Custom Monitoring

Add monitoring endpoints:

```bash
# Metrics endpoint
curl https://your-app.railway.app/metrics

# Status endpoint
curl https://your-app.railway.app/status
```

---

## 🎉 You're Ready!

Your MCP Meme Bot is now deployed on Railway with:

✅ **Automatic HTTPS**  
✅ **Custom domain support**  
✅ **Auto-scaling**  
✅ **Health monitoring**  
✅ **Enhanced security**  
✅ **Zero-downtime deployments**

Connect with:

```bash
/mcp connect https://your-app.railway.app/mcp Bearer your_auth_token
```

Enjoy your production-ready MCP Meme Bot! 🎭🚂

