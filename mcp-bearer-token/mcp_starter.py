import asyncio
import json
import random
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
from starlette.staticfiles import StaticFiles

import markdownify
import httpx
import readabilipy

# Import image utilities
from image_utils import (
    generate_image_with_gemini,
    overlay_caption,
    craft_captions,
    save_image,
    to_base64_data_url,
    natural_language_to_meme_params
)

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

# --- Load meme templates ---
def load_templates():
    """Load meme templates from the data directory."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the repo root, then into data folder
        data_path = os.path.join(os.path.dirname(script_dir), "data", "templates.json")
        with open(data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load templates: {e}")
        return []

TEMPLATES = load_templates()

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Static File Serving Setup ---
# Create generated directory and mount static files
generated_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
os.makedirs(generated_dir, exist_ok=True)

# Mount static files for serving generated images
app = mcp.app
app.mount("/media", StaticFiles(directory=generated_dir), name="media")

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Tool: meme_idea ---
MemeIdeaDescription = RichToolDescription(
    description="Suggest meme templates + captions for a given topic and optional mood",
    use_when="Use this to get creative meme ideas and template suggestions for any topic or mood",
    side_effects="Returns 5 meme template suggestions with example captions and preview URLs",
)

@mcp.tool(description=MemeIdeaDescription.model_dump_json())
async def meme_idea(
    topic: Annotated[str, Field(description="What is the meme about? E.g., Monday mornings")],
    mood: Annotated[str | None, Field(description="Optional mood or vibe, e.g., sarcastic, wholesome, dark")] = None,
) -> str:
    """
    Suggest meme templates and captions for a given topic and optional mood.
    """
    if not TEMPLATES:
        return "❌ Error: No meme templates available. Please check the data/templates.json file."
    
    # Convert inputs to lowercase for matching
    topic_lower = topic.lower()
    mood_lower = mood.lower() if mood else ""
    
    # Find matching templates
    matching_templates = []
    for template in TEMPLATES:
        template_tags = [tag.lower() for tag in template.get("tags", [])]
        
        # Check if topic or mood matches any tags
        topic_match = any(topic_lower in tag or tag in topic_lower for tag in template_tags)
        mood_match = any(mood_lower in tag or tag in mood_lower for tag in template_tags) if mood else False
        
        if topic_match or mood_match:
            matching_templates.append(template)
    
    # If we don't have enough matches, add random templates
    if len(matching_templates) < 5:
        remaining_templates = [t for t in TEMPLATES if t not in matching_templates]
        random.shuffle(remaining_templates)
        matching_templates.extend(remaining_templates[:5 - len(matching_templates)])
    
    # Take only the first 5
    selected_templates = matching_templates[:5]
    
    # Generate captions for each template
    def generate_captions(template_name, topic, mood):
        captions = []
        
        # Template-specific caption patterns
        if "distracted boyfriend" in template_name.lower():
            captions.extend([
                f"Me: {topic} | Also me: *gets distracted by {mood or 'something better' if mood else 'anything else'}*",
                f"Me trying to focus on {topic} | Me when {mood or 'something more interesting'} appears"
            ])
        elif "drake" in template_name.lower():
            captions.extend([
                f"Drake: *disapproves of {topic}* | Drake: *approves of {mood or 'the alternative'}*",
                f"Drake: *rejects {topic}* | Drake: *embraces {mood or 'better option'}*"
            ])
        elif "grumpy cat" in template_name.lower():
            captions.extend([
                f"Grumpy Cat: *exists* | Me on {topic}: *same energy*",
                f"Grumpy Cat: *judging {topic}* | Me: *feeling {mood or 'grumpy'}*"
            ])
        elif "success kid" in template_name.lower():
            captions.extend([
                f"Success Kid: *finally accomplishes {topic}* | Me: *feeling {mood or 'proud'}*",
                f"Success Kid: *nails {topic}* | Me: *celebrating {mood or 'victory'}*"
            ])
        else:
            # Generic captions
            captions.extend([
                f"When {topic} meets {mood or 'reality'}",
                f"The {mood or 'epic'} journey of {topic}",
                f"{topic}: *exists* | Me: *feels {mood or 'confused'}*"
            ])
        
        return captions[:3]  # Return max 3 captions
    
    # Build the response
    mood_display = f" & {mood}" if mood else ""
    response = f"🎯 **Topic**: {topic}{mood_display}\n\n"
    response += f"📝 **Found {len(selected_templates)} meme templates for you:**\n\n"
    
    for i, template in enumerate(selected_templates, 1):
        response += f"**{i}. {template['name']}**\n"
        response += f"🖼️ Preview: {template['preview']}\n"
        response += f"🏷️ Tags: {', '.join(template['tags'])}\n"
        
        captions = generate_captions(template['name'], topic, mood)
        response += "💬 Captions:\n"
        for caption in captions:
            response += f"• {caption}\n"
        response += "\n"
    
    # Add suggestions for generating real images
    response += f"🎨 **Want to generate a real meme image?**\n"
    response += f"**Structured**: `meme.generate topic=\"{topic}\" mood=\"{mood or ''}\" style=\"photo\" render_text=true`\n"
    response += f"**Natural**: `meme.generate.natural query=\"give me a {mood or 'funny'} meme about {topic}\"`\n"
    
    return response

# --- Tool: meme.generate ---
MemeGenerateDescription = RichToolDescription(
    description="Generate a custom meme image with Gemini, with optional top/bottom captions",
    use_when="Use this to create actual meme images from topics and moods using AI image generation",
    side_effects="Generates and saves a new meme image file, returns public URL and base64 data",
)

@mcp.tool(description=MemeGenerateDescription.model_dump_json())
async def meme_generate(
    topic: Annotated[str, Field(description="What is the meme about? E.g., Monday mornings")],
    mood: Annotated[str | None, Field(description="Optional mood or vibe, e.g., sarcastic, wholesome, dark")] = None,
    style: Annotated[str, Field(description="Image style: photo, cartoon, pixel, comic")] = "photo",
    render_text: Annotated[bool, Field(description="Whether to overlay captions on the image")] = True,
) -> str:
    """
    Generate a custom meme image using Google Gemini API.
    """
    try:
        # Build image prompt
        mood_desc = mood or "neutral"
        prompt = f"Create a meme-ready image about '{topic}' with a {mood_desc} vibe, style: {style}. Bold subject, centered framing, high contrast background with headroom for captions."
        
        # Generate image
        print(f"Generating image for topic: {topic}, mood: {mood}, style: {style}")
        image_bytes = generate_image_with_gemini(prompt)
        
        # Add captions if requested
        if render_text:
            top_caption, bottom_caption = craft_captions(topic, mood)
            image_bytes = overlay_caption(image_bytes, top_caption, bottom_caption)
        
        # Save image
        filename, filepath = save_image(image_bytes)
        
        # Create base64 data URL (truncated for message size)
        data_url = to_base64_data_url(image_bytes)
        truncated_data_url = data_url[:200] + "..." if len(data_url) > 200 else data_url
        
        # Build response
        response = f"Meme Generated\n"
        response += f"Topic: {topic}\n"
        response += f"Mood: {mood or '—'}\n"
        response += f"Style: {style}\n"
        response += f"Public URL: /media/{filename}\n\n"
        response += f"Data URL (fallback): {truncated_data_url}\n\n"
        response += f"Tip: If the image preview doesn't render inline, tap the Public URL."
        
        return response
        
    except RuntimeError as e:
        return f"Image generation failed: {str(e)}. Try again or change style."
    except Exception as e:
        print(f"Error in meme generation: {e}")
        return "Image model is busy or not available in your region. Try again or change style."

# --- Tool: meme.generate.natural ---
MemeGenerateNaturalDescription = RichToolDescription(
    description="Generate memes from natural language queries - just describe what you want!",
    use_when="Use this when users provide casual, natural language requests for memes instead of structured parameters",
    side_effects="Parses natural language and generates meme images with smart parameter detection",
)

@mcp.tool(description=MemeGenerateNaturalDescription.model_dump_json())
async def meme_generate_natural(
    query: Annotated[str, Field(description="Natural language description of the meme you want, e.g., 'give me a sarcastic meme about Monday mornings'")]
) -> str:
    """
    Generate a custom meme image from natural language description.
    
    Automatically detects topic, mood, style, and text preferences from casual text.
    """
    try:
        # Parse natural language into structured parameters
        params = natural_language_to_meme_params(query)
        
        # Extract parameters
        topic = params["topic"]
        mood = params["mood"]
        style = params["style"]
        render_text = params["render_text"]
        
        # Validate that we have a meaningful topic
        if not topic or len(topic.strip()) < 2:
            return f"I need more context about what kind of meme you want. Try something like 'funny meme about cats' or 'sarcastic office humor'."
        
        # Build image prompt
        mood_desc = mood or "neutral"
        prompt = f"Create a meme-ready image about '{topic}' with a {mood_desc} vibe, style: {style}. Bold subject, centered framing, high contrast background with headroom for captions."
        
        # Generate image
        print(f"Natural language query: '{query}'")
        print(f"Parsed -> Topic: '{topic}', Mood: {mood}, Style: {style}, Text: {render_text}")
        
        image_bytes = generate_image_with_gemini(prompt)
        
        # Add captions if requested
        if render_text:
            top_caption, bottom_caption = craft_captions(topic, mood)
            image_bytes = overlay_caption(image_bytes, top_caption, bottom_caption)
        
        # Save image
        filename, filepath = save_image(image_bytes)
        
        # Create base64 data URL (truncated for message size)
        data_url = to_base64_data_url(image_bytes)
        truncated_data_url = data_url[:200] + "..." if len(data_url) > 200 else data_url
        
        # Build response
        response = f"🎨 **Meme Generated from Natural Language**\n\n"
        response += f"**Your Request**: \"{query}\"\n\n"
        response += f"**Parsed Parameters**:\n"
        response += f"• Topic: {topic}\n"
        response += f"• Mood: {mood or '—'}\n"
        response += f"• Style: {style}\n"
        response += f"• Text Overlay: {'Yes' if render_text else 'No'}\n\n"
        response += f"**Result**: /media/{filename}\n\n"
        response += f"**Data URL (fallback)**: {truncated_data_url}\n\n"
        response += f"💡 **Tip**: If the image doesn't show, click the media URL above!"
        
        return response
        
    except RuntimeError as e:
        return f"Image generation failed: {str(e)}. Try rephrasing your request or change the style."
    except Exception as e:
        print(f"Error in natural language meme generation: {e}")
        return "Sorry, I couldn't process your meme request. Try being more specific about what you want."

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
