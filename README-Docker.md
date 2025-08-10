# 🐳 Docker Deployment Guide for MCP Meme Bot

This guide will help you deploy the MCP Meme Bot using Docker for easy, reliable deployment.

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Navigate to the project directory
cd meme-bot-mcp

# Start the entire stack (MCP server + Cloudflare tunnel)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f meme-bot-mcp
```

### Option 2: Using Docker Run Script

```bash
# Navigate to the project directory
cd meme-bot-mcp

# Make script executable (Linux/Mac)
chmod +x docker-run.sh

# Run the script
./docker-run.sh
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build -t meme-bot-mcp:latest .

# Run the container
docker run -d \
  --name meme-bot-mcp \
  --restart unless-stopped \
  -p 8086:8086 \
  -v "$(pwd)/logs:/app/logs" \
  meme-bot-mcp:latest
```

## 🔧 Configuration

### Environment Variables

You can customize the deployment using environment variables:

```bash
# Custom port
docker run -e MCP_PORT=9000 -p 9000:9000 meme-bot-mcp:latest

# Custom host
docker run -e MCP_HOST=127.0.0.1 meme-bot-mcp:latest
```

### Volumes

- `./logs:/app/logs` - Persist application logs
- Add more volumes as needed for data persistence

## 🌐 External Access with Cloudflare Tunnel

### Using Docker Compose (Included)

The docker-compose.yml includes a Cloudflare tunnel service:

```bash
# Start with tunnel
docker-compose up -d

# Check tunnel logs for the public URL
docker-compose logs cloudflare-tunnel
```

### Manual Tunnel Setup

```bash
# Run Cloudflare tunnel separately
docker run -d \
  --name cf-tunnel \
  --network host \
  cloudflare/cloudflared:latest \
  tunnel --url http://localhost:8086
```

## 🔍 Monitoring and Debugging

### Check Container Status

```bash
# List running containers
docker ps

# Check health status
docker inspect meme-bot-mcp | grep Health

# View detailed logs
docker logs -f meme-bot-mcp
```

### Health Checks

The container includes built-in health checks:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' meme-bot-mcp
```

## 🛠 Development

### Building for Development

```bash
# Build with development tags
docker build -t meme-bot-mcp:dev .

# Run with volume mounts for live development
docker run -d \
  --name meme-bot-mcp-dev \
  -p 8086:8086 \
  -v "$(pwd):/app" \
  meme-bot-mcp:dev
```

### Debugging

```bash
# Enter the container for debugging
docker exec -it meme-bot-mcp bash

# Check Python environment
docker exec meme-bot-mcp python --version

# Test API endpoint
docker exec meme-bot-mcp curl -f http://localhost:8086/mcp/
```

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**

   ```bash
   # Find what's using port 8086
   netstat -tulpn | grep 8086

   # Use different port
   docker run -p 8087:8086 meme-bot-mcp:latest
   ```

2. **Container Won't Start**

   ```bash
   # Check logs
   docker logs meme-bot-mcp

   # Check if image built correctly
   docker images | grep meme-bot-mcp
   ```

3. **Health Check Failing**
   ```bash
   # Check internal connectivity
   docker exec meme-bot-mcp curl -f http://localhost:8086/mcp/
   ```

## 🧹 Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (careful - this deletes data!)
docker-compose down -v

# Remove images
docker rmi meme-bot-mcp:latest

# Clean up everything
docker system prune -a
```

## 📦 Production Deployment

### Using a Registry

```bash
# Tag for registry
docker tag meme-bot-mcp:latest your-registry.com/meme-bot-mcp:latest

# Push to registry
docker push your-registry.com/meme-bot-mcp:latest

# Deploy on production server
docker pull your-registry.com/meme-bot-mcp:latest
docker run -d \
  --name meme-bot-mcp \
  --restart always \
  -p 8086:8086 \
  your-registry.com/meme-bot-mcp:latest
```

### Security Considerations

1. **Run as non-root user** (already configured)
2. **Use secrets for sensitive data**
3. **Configure firewall rules**
4. **Use HTTPS in production**
5. **Regular security updates**

## 🔗 MCP Connection

Once deployed, connect using:

```
/mcp connect https://your-tunnel-url.trycloudflare.com/mcp Bearer devtoken_shared_for_team
```

Or for local development:

```
/mcp connect http://localhost:8086/mcp Bearer devtoken_shared_for_team
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [FastMCP Documentation](https://gofastmcp.com/)

