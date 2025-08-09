import asyncio
from typing import Annotated, List
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import textwrap
import io
import base64
from functools import partial
import random
import time
import hmac
import hashlib
import json

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

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
    "Meme Bot",
    auth=SimpleBearerAuthProvider(TOKEN),
)

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

# --- Meme Generation Tool ---
GENERATE_MEME_DESCRIPTION = RichToolDescription(
    description="Generate memes in different styles (dank, classic, wholesome) with AI-powered captions.",
    use_when="Use this tool to create memes with custom text or get AI suggestions in specific meme styles.",
    side_effects="Returns the generated meme image with styled text overlay.",
)

# Meme style configurations
MEME_STYLES = {
    "dank": {
        "prompt": """You are a dank meme expert. Generate a dank meme caption that's edgy, ironic, and uses internet culture references. 
                    Format: TOP: <text>
                    BOTTOM: <text>
                    Make it absurd and unexpected, possibly using meme slang like 'bruh', 'ngl', 'fr fr', etc.
                    Keep each line under 50 characters.""",
        "font_size_factor": 0.12,  # Slightly larger text
        "outline_width": 2,
    },
    "classic": {
        "prompt": """Generate a classic meme caption in the style of early 2010s advice animals/image macros.
                    Format: TOP: <text>
                    BOTTOM: <text>
                    Use clear setup/punchline structure. Keep each line under 50 characters.""",
        "font_size_factor": 0.1,
        "outline_width": 1,
    },
    "wholesome": {
        "prompt": """Generate a wholesome, positive meme caption that's uplifting and kind.
                    Format: TOP: <text>
                    BOTTOM: <text>
                    Focus on positivity, self-care, friendship, or motivation. Keep each line under 50 characters.""",
        "font_size_factor": 0.1,
        "outline_width": 1,
    }
}

@mcp.tool(description=GENERATE_MEME_DESCRIPTION.model_dump_json())
async def generate_meme(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to use as meme template")] = None,
    top_text: Annotated[str | None, Field(description="Text to add at the top of the meme")] = None,
    bottom_text: Annotated[str | None, Field(description="Text to add at the bottom of the meme")] = None,
    suggest_text: Annotated[bool, Field(description="If true, use Gemini to suggest meme text")] = False,
    style: Annotated[str, Field(description="Meme style: 'dank', 'classic', or 'wholesome'")] = "classic",
    image_description: Annotated[str | None, Field(description="Optional description of the image for better AI suggestions")] = None,
) -> list[TextContent | ImageContent]:
    try:
        # Validate style
        if style not in MEME_STYLES:
            style = "classic"
        
        style_config = MEME_STYLES[style]
        
        # Decode and open image
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # If text suggestions requested, use Gemini
        if suggest_text:
            # Build prompt with image context if provided
            context = f"\nImage context: {image_description}" if image_description else ""
            prompt = style_config["prompt"] + context
            
            response = gemini_model.generate_content(prompt)
            text = response.text
            
            # Parse Gemini response
            lines = text.strip().split('\n')
            if len(lines) >= 2:
                top_text = lines[0].replace('TOP:', '').strip()
                bottom_text = lines[1].replace('BOTTOM:', '').strip()

        # Create copy for drawing
        meme = image.copy()
        draw = ImageDraw.Draw(meme)
        
        # Calculate font size based on image size and style
        font_size = int(meme.width * style_config["font_size_factor"])
        try:
            # Try to load Impact font first (classic meme font)
            try:
                font = ImageFont.truetype("impact.ttf", font_size)
            except:
                font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

        # Helper function to draw outlined text with style-specific settings
        def draw_outlined_text(text, position, font):
            outline_width = style_config["outline_width"]
            # Draw black outline with style-specific width
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:  # Skip the center position
                        draw.text((position[0]+dx, position[1]+dy), text, font=font, fill='black')
            # Draw white text
            draw.text(position, text, font=font, fill='white')

        # Add top text if provided
        if top_text:
            # Wrap text to fit image width
            wrapped_text = textwrap.fill(top_text, width=20)
            text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text at top center
            x = (meme.width - text_width) // 2
            y = 10
            draw_outlined_text(wrapped_text, (x, y), font)

        # Add bottom text if provided
        if bottom_text:
            # Wrap text to fit image width
            wrapped_text = textwrap.fill(bottom_text, width=20)
            text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text at bottom center
            x = (meme.width - text_width) // 2
            y = meme.height - text_height - 10
            draw_outlined_text(wrapped_text, (x, y), font)

        # Convert to base64
        buf = io.BytesIO()
        meme.save(buf, format="PNG")
        meme_bytes = buf.getvalue()
        meme_base64 = base64.b64encode(meme_bytes).decode("utf-8")

        # Return both the image and text description
        return [
            TextContent(type="text", text=f"Generated meme with text:\nTop: {top_text or 'None'}\nBottom: {bottom_text or 'None'}"),
            ImageContent(type="image", mimeType="image/png", data=meme_base64)
        ]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to generate meme: {str(e)}"))

# --- Meme Templates Tool ---
GET_MEME_TEMPLATES_DESCRIPTION = RichToolDescription(
    description="Fetch meme templates (latest/popular) and return them as images you can caption.",
    use_when="Use when the user asks for meme templates by name or wants the latest/popular templates.",
    side_effects="Downloads remote images and returns them inline as PNGs.",
)


async def _download_and_png(client: httpx.AsyncClient, url: str) -> bytes:
    resp = await client.get(url, follow_redirects=True, timeout=30)
    if resp.status_code >= 400:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to download template image: {url} ({resp.status_code})"))
    try:
        original = Image.open(io.BytesIO(resp.content))
        buf = io.BytesIO()
        original.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # If Pillow fails, fall back to raw bytes assuming it's already a PNG/JPEG
        return resp.content


@mcp.tool(description=GET_MEME_TEMPLATES_DESCRIPTION.model_dump_json())
async def get_meme_templates(
    source: Annotated[str, Field(description="Source for templates: 'imgflip' or 'memegen'")] = "imgflip",
    limit: Annotated[int, Field(description="How many templates to return (max 20)")] = 6,
    search: Annotated[str | None, Field(description="Optional search filter by name/id")] = None,
) -> list[TextContent | ImageContent]:
    """Return a set of meme template images from popular sources."""
    try:
        limit = max(1, min(limit, 20))
        templates: list[dict] = []

        async with httpx.AsyncClient() as client:
            if source.lower() == "memegen":
                # https://api.memegen.link/templates/
                r = await client.get("https://api.memegen.link/templates/", timeout=30)
                if r.status_code != 200:
                    raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch templates from memegen ({r.status_code})"))
                data = r.json()
                # Each item has: id, name, blank, example
                for item in data:
                    name = item.get("name") or item.get("id") or ""
                    if search and search.lower() not in str(name).lower():
                        continue
                    templates.append({
                        "id": item.get("id"),
                        "name": name,
                        "image_url": item.get("blank") or (item.get("example") or {}).get("url"),
                        "source": "memegen",
                    })
            else:
                # Default to imgflip popular templates
                r = await client.get("https://api.imgflip.com/get_memes", timeout=30)
                if r.status_code != 200:
                    raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch templates from imgflip ({r.status_code})"))
                payload = r.json()
                memes = (payload.get("data") or {}).get("memes") or []
                for m in memes:
                    name = m.get("name", "")
                    if search and search.lower() not in name.lower():
                        continue
                    templates.append({
                        "id": str(m.get("id")),
                        "name": name,
                        "image_url": m.get("url"),
                        "source": "imgflip",
                    })

            # Cap and download images
            selected = templates[:limit]

            async def fetch_one(tpl: dict) -> tuple[dict, bytes | None]:
                url = tpl.get("image_url")
                if not url:
                    return tpl, None
                try:
                    img_bytes = await _download_and_png(client, url)
                    return tpl, img_bytes
                except Exception:
                    return tpl, None

            results = await asyncio.gather(*[fetch_one(t) for t in selected])

        # Build response contents
        contents: list[TextContent | ImageContent] = []
        if not results:
            return [TextContent(type="text", text="No templates found for the given query/source.")]

        summary_lines = [
            f"Source: {source}",
            f"Returned: {len(results)} templates",
        ]
        for tpl, img_bytes in results:
            summary_lines.append(f"- {tpl.get('name')} (id: {tpl.get('id')})")
            if img_bytes:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                contents.append(ImageContent(type="image", mimeType="image/png", data=b64))

        contents.insert(0, TextContent(type="text", text="\n".join(summary_lines)))
        return contents
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch meme templates: {str(e)}"))

# --- Caption Existing Template (no AI) ---
CAPTION_MEME_TEMPLATE_DESCRIPTION = RichToolDescription(
    description="Add top/bottom text to a provided meme image in a chosen style (no AI).",
    use_when="Use when you already have a meme template image and want to overlay custom text.",
    side_effects="Returns the captioned meme image as PNG.",
)


@mcp.tool(description=CAPTION_MEME_TEMPLATE_DESCRIPTION.model_dump_json())
async def caption_meme_template(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data (template to caption)")],
    top_text: Annotated[str | None, Field(description="Top caption text")] = None,
    bottom_text: Annotated[str | None, Field(description="Bottom caption text")] = None,
    style: Annotated[str, Field(description="Meme style for font/outline: 'dank', 'classic', 'wholesome'")] = "classic",
) -> list[TextContent | ImageContent]:
    try:
        if not top_text and not bottom_text:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide at least one of top_text or bottom_text."))

        if style not in MEME_STYLES:
            style = "classic"
        style_config = MEME_STYLES[style]

        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        meme = image.copy()
        draw = ImageDraw.Draw(meme)

        font_size = int(meme.width * style_config["font_size_factor"])
        try:
            try:
                font = ImageFont.truetype("impact.ttf", font_size)
            except Exception:
                font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        def draw_outlined_text(text: str, position: tuple[int, int], font: ImageFont.FreeTypeFont):
            outline_width = style_config["outline_width"]
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((position[0] + dx, position[1] + dy), text, font=font, fill="black")
            draw.text(position, text, font=font, fill="white")

        if top_text:
            wrapped = textwrap.fill(top_text, width=20)
            bbox = draw.textbbox((0, 0), wrapped, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (meme.width - tw) // 2
            y = 10
            draw_outlined_text(wrapped, (x, y), font)

        if bottom_text:
            wrapped = textwrap.fill(bottom_text, width=20)
            bbox = draw.textbbox((0, 0), wrapped, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (meme.width - tw) // 2
            y = meme.height - th - 10
            draw_outlined_text(wrapped, (x, y), font)

        buf = io.BytesIO()
        meme.save(buf, format="PNG")
        out_bytes = buf.getvalue()
        out_b64 = base64.b64encode(out_bytes).decode("utf-8")

        return [
            TextContent(type="text", text=f"Captioned meme created (style={style})."),
            ImageContent(type="image", mimeType="image/png", data=out_b64),
        ]
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to caption meme: {str(e)}"))


# --- Guess the Meme Quiz ---
MEME_QUIZ_DESCRIPTION = RichToolDescription(
    description="Start a 'guess the meme' quiz: returns a template image and 3 name options.",
    use_when="Use to quiz the user on identifying meme templates.",
    side_effects="Downloads an image and returns a question token to validate answers.",
)


def _quiz_token_create(secret: str, correct_name: str) -> str:
    ts = int(time.time())
    msg = f"{correct_name}|{ts}"
    sig = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
    token = base64.b64encode(f"{correct_name}|{ts}|{sig}".encode("utf-8")).decode("utf-8")
    return token


def _quiz_token_verify(secret: str, token: str) -> tuple[bool, str]:
    try:
        raw = base64.b64decode(token.encode("utf-8")).decode("utf-8")
        parts = raw.split("|")
        if len(parts) != 3:
            return False, ""
        name, ts, sig = parts
        msg = f"{name}|{ts}"
        exp_sig = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, exp_sig):
            return True, name
        return False, ""
    except Exception:
        return False, ""


async def _fetch_imgflip_templates(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get("https://api.imgflip.com/get_memes", timeout=30)
    if r.status_code != 200:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch templates from imgflip ({r.status_code})"))
    payload = r.json()
    memes = (payload.get("data") or {}).get("memes") or []
    results = []
    for m in memes:
        results.append({
            "id": str(m.get("id")),
            "name": m.get("name", ""),
            "image_url": m.get("url"),
        })
    return results


@mcp.tool(description=MEME_QUIZ_DESCRIPTION.model_dump_json())
async def get_meme_quiz_question(
    source: Annotated[str, Field(description="Template source: 'imgflip' (default) or 'memegen'")] = "imgflip",
) -> list[TextContent | ImageContent]:
    try:
        async with httpx.AsyncClient() as client:
            if source.lower() == "memegen":
                r = await client.get("https://api.memegen.link/templates/", timeout=30)
                if r.status_code != 200:
                    raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch templates from memegen ({r.status_code})"))
                data = r.json()
                candidates = [{
                    "id": it.get("id"),
                    "name": it.get("name") or it.get("id") or "",
                    "image_url": it.get("blank") or (it.get("example") or {}).get("url"),
                } for it in data]
            else:
                candidates = await _fetch_imgflip_templates(client)

            # Need at least 3 distinct options
            random.shuffle(candidates)
            unique = []
            seen = set()
            for c in candidates:
                nm = c.get("name") or ""
                if nm and nm not in seen and c.get("image_url"):
                    unique.append(c)
                    seen.add(nm)
                if len(unique) >= 10:
                    break
            if len(unique) < 3:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="Not enough templates to build a quiz question."))

            picks = random.sample(unique, 3)
            correct = picks[0]
            options = [p["name"] for p in picks]
            random.shuffle(options)

            # Download correct image
            img_bytes = await _download_and_png(client, correct["image_url"])
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            token = _quiz_token_create(TOKEN, correct["name"])
            options_text = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])

            return [
                TextContent(type="text", text=(
                    "Guess the meme template!\n" +
                    options_text +
                    f"\nquestion_id: {token}\nReply using tool answer_meme_quiz with question_id and your choice index (1-3)."
                )),
                ImageContent(type="image", mimeType="image/png", data=img_b64),
            ]
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to create quiz question: {str(e)}"))


ANSWER_MEME_QUIZ_DESCRIPTION = RichToolDescription(
    description="Submit an answer to a 'guess the meme' question and get correctness feedback.",
    use_when="Use after get_meme_quiz_question returns question_id and options.",
)


@mcp.tool(description=ANSWER_MEME_QUIZ_DESCRIPTION.model_dump_json())
async def answer_meme_quiz(
    question_id: Annotated[str, Field(description="The question_id token returned by get_meme_quiz_question")],
    choice_index: Annotated[int | None, Field(description="Your selected option index (1-3)")] = None,
    answer_text: Annotated[str | None, Field(description="Alternatively provide the template name directly")] = None,
) -> str:
    try:
        ok, correct_name = _quiz_token_verify(TOKEN, question_id)
        if not ok:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid or expired question_id."))

        if choice_index is None and not answer_text:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide choice_index (1-3) or answer_text."))

        if answer_text:
            given = answer_text.strip()
        else:
            given = f"{choice_index}"  # Let the client map index to option text if needed

        # Accept either exact name match or index-based clients can pass the name via answer_text
        if answer_text and given.lower() == correct_name.lower():
            return f"Correct! The template was: {correct_name}"
        else:
            return f"Your answer: {given}. The correct template was: {correct_name}"
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to evaluate quiz answer: {str(e)}"))


# --- Multi-question Quiz Sessions (5 Qs + roast result) ---
QUIZ_SESSIONS: dict[str, dict] = {}


async def _fetch_memegen_templates(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get("https://api.memegen.link/templates/", timeout=30)
    if r.status_code != 200:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch templates from memegen ({r.status_code})"))
    data = r.json()
    results = []
    for item in data:
        results.append({
            "id": item.get("id"),
            "name": item.get("name") or item.get("id") or "",
            "image_url": item.get("blank") or (item.get("example") or {}).get("url"),
        })
    return results


async def _build_quiz_questions(client: httpx.AsyncClient, source: str, total_questions: int) -> list[dict]:
    source_lower = source.lower()
    if source_lower == "memegen":
        candidates = await _fetch_memegen_templates(client)
    else:
        candidates = await _fetch_imgflip_templates(client)

    # Unique templates with valid name and image
    candidates = [c for c in candidates if (c.get("name") and c.get("image_url"))]
    random.shuffle(candidates)
    if len(candidates) < max(6, total_questions + 2):
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Not enough templates to create a full quiz."))

    selected = candidates[: total_questions]
    questions: list[dict] = []
    for correct_tpl in selected:
        # pick two distractors with different names
        distractor_pool = [c for c in candidates if c["name"] != correct_tpl["name"]]
        distractors = random.sample(distractor_pool, 2)

        options = [correct_tpl["name"], distractors[0]["name"], distractors[1]["name"]]
        random.shuffle(options)

        # Download image bytes now to freeze the question
        img_bytes = await _download_and_png(client, correct_tpl["image_url"])
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        questions.append({
            "correct": correct_tpl["name"],
            "options": options,
            "image_b64": img_b64,
        })
    return questions


def _make_session_id(secret: str) -> str:
    nonce = os.urandom(16).hex()
    ts = str(int(time.time()))
    msg = f"{nonce}|{ts}"
    sig = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
    return base64.b64encode(f"{nonce}|{ts}|{sig}".encode("utf-8")).decode("utf-8")


def _choose_roast_text(score: int, total: int) -> tuple[str, str]:
    bracket = score / max(1, total)
    if bracket >= 0.8:
        style = "dank"
        top = random.choice(["OK QUIZ GOD", "BIG BRAIN ENERGY", "SPEEDRUNNING MEMES"])
        bottom = random.choice(["RESPECT", "TEACH ME SENPAI", "CARRY US PLS"])
    elif bracket >= 0.4:
        style = "classic"
        top = random.choice(["NOT BAD", "MID BUT TRYING", "HALF RIGHT HERO"])
        bottom = random.choice(["COULD BE WORSE", "KEEP GRINDING", "ALMOST THERE"])
    else:
        style = "dank"
        top = random.choice(["BRO...", "SKULL EMOJI", "SEND HELP"])
        bottom = random.choice(["MEME LITERACY = 0", "TOUCH GRASS", "UNINSTALL INTERNET"])
    return style, f"{top}\n{bottom}"


def _render_options_on_image(image_bytes: bytes, options: list[str]) -> bytes:
    """Overlay the 3 choice options on top of the image so clients always show them."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(image)

    # Draw translucent panel at bottom for text readability
    panel_height = max(60, int(image.height * 0.22))
    panel = Image.new("RGBA", (image.width, panel_height), (0, 0, 0, 160))
    image.alpha_composite(panel, (0, image.height - panel_height))

    # Font sizing
    font_size = max(18, int(image.width * 0.045))
    try:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    lines = [f"{i+1}) {opt}" for i, opt in enumerate(options)]
    y = image.height - panel_height + 10
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = max(12, (image.width - text_width) // 2)
        # Outline
        for dx in (-1, 1):
            for dy in (-1, 1):
                draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += font_size + 6

    out = io.BytesIO()
    image.convert("RGB").save(out, format="PNG")
    return out.getvalue()


@mcp.tool(description=RichToolDescription(
    description="Start a 5-question meme quiz session and get question 1.",
    use_when="User wants to play a meme guess quiz with score and final roast reward.",
).model_dump_json())
async def start_meme_quiz(
    source: Annotated[str, Field(description="Template source: 'imgflip' or 'memegen'")] = "imgflip",
    num_questions: Annotated[int, Field(description="Number of questions (default 5, max 10)")] = 5,
) -> list[TextContent | ImageContent]:
    try:
        total_questions = max(1, min(num_questions, 10))
        async with httpx.AsyncClient() as client:
            questions = await _build_quiz_questions(client, source, total_questions)

        session_id = _make_session_id(TOKEN)
        QUIZ_SESSIONS[session_id] = {
            "source": source,
            "questions": questions,
            "index": 0,
            "score": 0,
            "created_at": time.time(),
        }

        q = questions[0]
        # Compose image with options overlaid
        base_img_bytes = base64.b64decode(q["image_b64"]) if isinstance(q["image_b64"], str) else q["image_b64"]
        composed = _render_options_on_image(base_img_bytes, q["options"])
        # Start timer for this question
        QUIZ_SESSIONS[session_id]["question_start_ts"] = time.time()

        options_text = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(q["options"])])
        header = (
            "Meme Quiz — Question 1/" + str(total_questions) + "\n" +
            options_text + f"\nsession_id: {session_id}\nReply using tool answer_meme_quiz_session with session_id and your choice index (1-3).\nTime limit: 10 seconds"
        )
        return [
            TextContent(type="text", text=header),
            ImageContent(type="image", mimeType="image/png", data=base64.b64encode(composed).decode("utf-8")),
        ]
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to start quiz: {str(e)}"))


@mcp.tool(description=RichToolDescription(
    description="Answer a quiz question by session_id. Returns feedback and next question or final roast reward.",
    use_when="Continuing an active meme quiz session.",
).model_dump_json())
async def answer_meme_quiz_session(
    session_id: Annotated[str, Field(description="The session_id returned by start_meme_quiz")],
    choice_index: Annotated[int | None, Field(description="Your selected option index (1-3)")] = None,
    answer_text: Annotated[str | None, Field(description="Alternatively provide the template name directly")] = None,
) -> list[TextContent | ImageContent]:
    try:
        state = QUIZ_SESSIONS.get(session_id)
        if not state:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Unknown or expired session_id. Start a new quiz."))

        questions: list[dict] = state["questions"]
        idx: int = state["index"]
        score: int = state["score"]
        total = len(questions)
        if idx >= total:
            # Already finished; provide summary again
            style, roast_text = _choose_roast_text(score, total)
            return [TextContent(type="text", text=f"Quiz already completed. Score: {score}/{total}. Roast: {roast_text}")]

        current = questions[idx]
        correct_name = current["correct"]

        # Enforce 10-second timer
        start_ts = state.get("question_start_ts")
        timed_out = False
        if isinstance(start_ts, (int, float)):
            if (time.time() - float(start_ts)) > 10.0:
                timed_out = True

        if answer_text:
            given_ok = (answer_text.strip().lower() == correct_name.lower())
        else:
            # If only index provided, we cannot know which option text was chosen on server
            # so we consider it incorrect unless client supplies text; but we can still accept index
            try:
                given_ok = (1 <= (choice_index or 0) <= 3 and current["options"][int(choice_index) - 1].lower() == correct_name.lower())
            except Exception:
                given_ok = False

        if timed_out:
            feedback = f"Time's up! Correct was: {correct_name} ({score}/{total})"
        elif given_ok:
            score += 1
            state["score"] = score
            feedback = f"Correct! ({score}/{total})"
        else:
            feedback = f"Wrong. Correct was: {correct_name} ({score}/{total})"

        # Move to next question
        idx += 1
        state["index"] = idx

        if idx < total:
            q = questions[idx]
            options_text = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(q["options"])])
            # Compose next image with options and reset timer
            base_img_bytes_next = base64.b64decode(q["image_b64"]) if isinstance(q["image_b64"], str) else q["image_b64"]
            composed_next = _render_options_on_image(base_img_bytes_next, q["options"]) 
            state["question_start_ts"] = time.time()
            header = (
                feedback + "\n\n" +
                f"Meme Quiz — Question {idx+1}/{total}\n" +
                options_text + f"\nsession_id: {session_id}\nReply again with your choice.\nTime limit: 10 seconds"
            )
            return [
                TextContent(type="text", text=header),
                ImageContent(type="image", mimeType="image/png", data=base64.b64encode(composed_next).decode("utf-8")),
            ]
        else:
            # Finished — return roast meme reward
            style, roast_combined = _choose_roast_text(score, total)
            top, bottom = roast_combined.split("\n", 1) if "\n" in roast_combined else (roast_combined, "")

            # Pick a random template for the roast image
            async with httpx.AsyncClient() as client:
                pool = await _fetch_imgflip_templates(client)
                pool = [p for p in pool if p.get("image_url")]
                chosen = random.choice(pool)
                roast_bytes = await _download_and_png(client, chosen["image_url"])
            # Draw text
            image = Image.open(io.BytesIO(roast_bytes))
            meme = image.copy()
            draw = ImageDraw.Draw(meme)
            font_size = int(meme.width * MEME_STYLES[style]["font_size_factor"])
            try:
                try:
                    font = ImageFont.truetype("impact.ttf", font_size)
                except Exception:
                    font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            def draw_outlined(text: str, pos: tuple[int, int]):
                ow = MEME_STYLES[style]["outline_width"]
                for dx in range(-ow, ow + 1):
                    for dy in range(-ow, ow + 1):
                        if dx != 0 or dy != 0:
                            draw.text((pos[0] + dx, pos[1] + dy), text, font=font, fill="black")
                draw.text(pos, text, font=font, fill="white")

            if top:
                wrapped = textwrap.fill(top, width=20)
                bbox = draw.textbbox((0, 0), wrapped, font=font)
                x = (meme.width - (bbox[2] - bbox[0])) // 2
                draw_outlined(wrapped, (x, 10))
            if bottom:
                wrapped = textwrap.fill(bottom, width=20)
                bbox = draw.textbbox((0, 0), wrapped, font=font)
                x = (meme.width - (bbox[2] - bbox[0])) // 2
                y = meme.height - ((bbox[3] - bbox[1])) - 10
                draw_outlined(wrapped, (x, y))

            buf = io.BytesIO()
            meme.save(buf, format="PNG")
            reward_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Cleanup session
            QUIZ_SESSIONS.pop(session_id, None)

            summary = f"Quiz finished! Score: {score}/{total}. Enjoy your roast reward."
            return [
                TextContent(type="text", text=summary),
                ImageContent(type="image", mimeType="image/png", data=reward_b64),
            ]
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to process quiz answer: {str(e)}"))

# --- Run MCP Server ---
async def main():
    try:
        print("🚀 Starting MCP server on http://0.0.0.0:8086")
    except Exception:
        # Fallback for terminals without UTF-8
        print("Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
