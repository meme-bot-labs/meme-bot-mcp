# MCP Starter for Puch AI

This is a starter template for creating your own Model Context Protocol (MCP) server that works with Puch AI. It comes with ready-to-use tools for job searching and image processing.

## What is MCP?

MCP (Model Context Protocol) allows AI assistants like Puch to connect to external tools and data sources safely. Think of it like giving your AI extra superpowers without compromising security.

## What's Included in This Starter?

### 🎯 Job Finder Tool
- **Analyze job descriptions** - Paste any job description and get smart insights
- **Fetch job postings from URLs** - Give a job posting link and get the full details
- **Search for jobs** - Use natural language to find relevant job opportunities

### 🖼️ Image Processing Tool
- **Convert images to black & white** - Upload any image and get a monochrome version

### 🎭 Meme Idea Tool
- **Get meme template suggestions** - Provide a topic and mood to get 5 meme templates with captions
- **Smart template matching** - Finds templates that match your topic and mood
- **Creative captions** - Generates funny captions for each template

### 🎨 Meme Generation Tool (Gemini)
- **Generate real meme images** - Create custom memes using Google Gemini AI
- **Multiple styles** - Choose from photo, cartoon, pixel, or comic styles
- **Automatic captions** - Add classic meme text overlays
- **Public URLs** - Get shareable links to your generated memes

### 🔐 Built-in Authentication
- Bearer token authentication (required by Puch AI)
- Validation tool that returns your phone number

## Quick Setup Guide

### Step 1: Install Dependencies

First, make sure you have Python 3.11 or higher installed. Then:

```bash
# Create virtual environment
uv venv

# Install all required packages
uv sync

# Activate the environment
source .venv/bin/activate
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Then edit `.env` and add your details:

```env
AUTH_TOKEN=your_secret_token_here
MY_NUMBER=919876543210

# Gemini API Configuration (for meme generation)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_IMAGE_MODEL=gemini-2.0-flash-exp
```

**Important Notes:**
- `AUTH_TOKEN`: This is your secret token for authentication. Keep it safe!
- `MY_NUMBER`: Your WhatsApp number in format `{country_code}{number}` (e.g., `919876543210` for +91-9876543210)
- `GEMINI_API_KEY`: Get this from [Google AI Studio](https://aistudio.google.com/app/apikey)
- `GEMINI_IMAGE_MODEL`: Default model for image generation (can be changed later)

### Step 3: Run the Server

```bash
cd mcp-bearer-token
python mcp_starter.py
```

You'll see: `🚀 Starting MCP server on http://0.0.0.0:8086`

### Step 4: Make It Public (Required by Puch)

Since Puch needs to access your server over HTTPS, you need to expose your local server:

#### Option A: Using ngrok (Recommended)

1. **Install ngrok:**
   Download from https://ngrok.com/download

2. **Get your authtoken:**
   - Go to https://dashboard.ngrok.com/get-started/your-authtoken
   - Copy your authtoken
   - Run: `ngrok config add-authtoken YOUR_AUTHTOKEN`

3. **Start the tunnel:**
   ```bash
   ngrok http 8086
   ```

#### Option B: Deploy to Cloud

You can also deploy this to services like:
- Railway
- Render
- Heroku
- DigitalOcean App Platform

## How to Connect with Puch AI

1. **[Open Puch AI](https://wa.me/+919998881729)** in your browser
2. **Start a new conversation**
3. **Use the connect command:**
   ```
   /mcp connect https://your-domain.ngrok.app/mcp your_secret_token_here
   ```

### Debug Mode

To get more detailed error messages:

```
/mcp diagnostics-level debug
```

## Customizing the Starter

### Adding New Tools

1. **Create a new tool function:**
   ```python
   @mcp.tool(description="Your tool description")
   async def your_tool_name(
       parameter: Annotated[str, Field(description="Parameter description")]
   ) -> str:
       # Your tool logic here
       return "Tool result"
   ```

2. **Add required imports** if needed


## 📚 **Additional Documentation Resources**

### **Official Puch AI MCP Documentation**
- **Main Documentation**: https://puch.ai/mcp
- **Protocol Compatibility**: Core MCP specification with Bearer & OAuth support
- **Command Reference**: Complete MCP command documentation
- **Server Requirements**: Tool registration, validation, HTTPS requirements

### **Technical Specifications**
- **JSON-RPC 2.0 Specification**: https://www.jsonrpc.org/specification (for error handling)
- **MCP Protocol**: Core protocol messages, tool definitions, authentication

### **Supported vs Unsupported Features**

**✓ Supported:**
- Core protocol messages
- Tool definitions and calls
- Authentication (Bearer & OAuth)
- Error handling

**✗ Not Supported:**
- Videos extension
- Resources extension
- Prompts extension

## Getting Help

- **Join Puch AI Discord:** https://discord.gg/VMCnMvYx
- **Check Puch AI MCP docs:** https://puch.ai/mcp
- **Puch WhatsApp Number:** +91 99988 81729

## Testing meme.idea Tool

After connecting to your MCP server with `/mcp connect` or `/mcp use`, you can test the meme idea tool:

```
meme.idea topic="Monday mornings" mood="sarcastic"
```

This will return 5 meme template suggestions with:
- Template names and preview URLs
- Relevant tags
- 2-3 creative captions for each template

### Example Usage:
- `meme.idea topic="work from home" mood="lazy"`
- `meme.idea topic="coffee" mood="addicted"`
- `meme.idea topic="programming" mood="frustrated"`

## Generating Real Memes (Gemini)

After connecting to your MCP server, you can generate actual meme images using Google Gemini AI:

### Structured Parameters
```
meme.generate topic="Monday mornings" mood="sarcastic" style="photo" render_text=true
meme.generate topic="Sunday holiday" mood="happy" style="cartoon" render_text=true
meme.generate topic="Relationship goals" mood="wholesome" style="comic" render_text=true
```

### Natural Language (New!)
Just describe what you want in plain English:
```
meme.generate.natural query="give me a sarcastic meme about my boss on office mondays"
meme.generate.natural query="create a funny cartoon meme about cats being dramatic"
meme.generate.natural query="make a wholesome meme about weekend plans"
meme.generate.natural query="show me an angry meme about traffic without text"
```

### Parameters:
- `topic`: What the meme is about (required)
- `mood`: Optional mood or vibe (e.g., sarcastic, wholesome, dark)
- `style`: Image style - "photo", "cartoon", "pixel", or "comic" (default: "photo")
- `render_text`: Whether to add meme captions (default: true)

### Output:
- **Public URL**: `/media/<filename>.png` - Accessible via your tunnel
- **Data URL**: Base64 encoded image (truncated for message size)
- **Saved locally**: Images are stored in the `generated/` folder

**Note**: All generated images include SynthID watermark per Google policy.

---

**Happy coding! 🚀**

Use the hashtag `#BuildWithPuch` in your posts about your MCP!

This starter makes it super easy to create your own MCP server for Puch AI. Just follow the setup steps and you'll be ready to extend Puch with your custom tools!
