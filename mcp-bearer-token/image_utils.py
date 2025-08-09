"""
Image utilities for meme generation using Google Gemini API.
"""
import os
import uuid
import base64
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Try to import google-genai, handle gracefully if not available
try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

def generate_image_with_gemini(prompt: str) -> bytes:
    """
    Generate an image using Google Gemini API.
    
    Args:
        prompt: The image generation prompt
        
    Returns:
        PNG image bytes
        
    Raises:
        RuntimeError: If generation fails
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai package not installed. Please install it with: pip install google-genai")
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-exp")
        
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment")
        
        # Initialize Gemini client
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Generate image with timeout
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="image/png"
            )
        )
        
        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError("No image generated")
        
        # Extract image data
        image_data = response.candidates[0].content.parts[0].inline_data.data
        return base64.b64decode(image_data)
        
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")

def overlay_caption(image_bytes: bytes, top: Optional[str], bottom: Optional[str]) -> bytes:
    """
    Overlay meme captions on an image.
    
    Args:
        image_bytes: Input image bytes
        top: Top caption text
        bottom: Bottom caption text
        
    Returns:
        PNG image bytes with captions
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default
        try:
            font_size = max(20, image.width // 20)
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        def draw_caption(text: str, position: str):
            if not text:
                return
                
            # Convert to uppercase and wrap text
            text = text.upper()
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] < image.width * 0.9:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw each line
            y_offset = 20 if position == "top" else image.height - len(lines) * font_size - 20
            
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (image.width - text_width) // 2
                
                # Draw black outline
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        draw.text((x + dx, y_offset + dy), line, font=font, fill="black")
                
                # Draw white text
                draw.text((x, y_offset), line, font=font, fill="white")
                y_offset += font_size + 5
        
        # Draw captions
        draw_caption(top, "top")
        draw_caption(bottom, "bottom")
        
        # Convert to bytes
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
        
    except Exception as e:
        # Return original image if caption overlay fails
        return image_bytes

def craft_captions(topic: str, mood: Optional[str]) -> Tuple[str, str]:
    """
    Create meme-style captions.
    
    Args:
        topic: The meme topic
        mood: Optional mood
        
    Returns:
        Tuple of (top_caption, bottom_caption)
    """
    import random
    
    topic_upper = topic.upper()
    mood_upper = mood.upper() if mood else "REALITY"
    
    templates = [
        (f"WHEN {topic_upper} MEETS {mood_upper}", "I'M TOTALLY FINE 🙂"),
        (f"{topic_upper} VIBES", f"{mood_upper} MODE ACTIVATED"),
        (f"{topic_upper}", f"STILL {mood_upper}"),
        (f"{topic_upper} BE LIKE", f"{mood_upper} ENERGY"),
        (f"{topic_upper} MOMENT", f"{mood_upper} FEELINGS"),
    ]
    
    if not mood:
        templates = [
            (f"{topic_upper}", "CAN'T RELATE"),
            (f"{topic_upper} VIBES", "MOOD"),
            (f"{topic_upper} MOMENT", "SAME"),
            (f"{topic_upper} BE LIKE", "FACTS"),
        ]
    
    return random.choice(templates)

def save_image(image_bytes: bytes) -> Tuple[str, str]:
    """
    Save image to generated folder.
    
    Args:
        image_bytes: Image data
        
    Returns:
        Tuple of (filename, absolute_path)
    """
    # Create generated directory
    generated_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(generated_dir, filename)
    
    # Save image
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    return filename, filepath

def to_base64_data_url(image_bytes: bytes) -> str:
    """
    Convert image bytes to base64 data URL.
    
    Args:
        image_bytes: Image data
        
    Returns:
        Base64 data URL
    """
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"

def natural_language_to_meme_params(user_query: str) -> dict:
    """
    Parse natural language query into meme generation parameters.
    
    Args:
        user_query: Natural language query from user
        
    Returns:
        Dictionary with topic, mood, style, render_text parameters
    """
    # Default parameters
    params = {
        "topic": user_query.strip(),
        "mood": None,
        "style": "photo",
        "render_text": True
    }
    
    # Convert to lowercase for matching
    query_lower = user_query.lower()
    
    # Detect mood keywords
    moods = ["funny", "sarcastic", "angry", "happy", "romantic", "sad", "wholesome", "dark", "silly", "excited"]
    for mood in moods:
        if mood in query_lower:
            params["mood"] = mood
            # Remove mood word from topic
            params["topic"] = user_query.replace(mood, "", 1).strip()
            break
    
    # Detect style keywords
    styles = ["photo", "cartoon", "pixel", "comic", "real"]
    for style in styles:
        if style in query_lower:
            params["style"] = style if style != "real" else "photo"
            # Remove style word from topic
            params["topic"] = params["topic"].replace(style, "", 1).strip()
            break
    
    # Check for render_text preferences
    no_text_keywords = ["no text", "without text", "no captions", "without captions", "just image", "image only"]
    for keyword in no_text_keywords:
        if keyword in query_lower:
            params["render_text"] = False
            # Remove the keyword from topic
            params["topic"] = params["topic"].replace(keyword, "", 1).strip()
            break
    
    # Clean up topic
    params["topic"] = params["topic"].strip()
    
    # Handle common prefixes like "give me", "create", "make", etc.
    prefixes_to_remove = [
        "give me", "create", "make", "generate", "show me", "i want", "i need",
        "can you make", "can you create", "please make", "please create"
    ]
    
    topic_lower = params["topic"].lower()
    for prefix in prefixes_to_remove:
        if topic_lower.startswith(prefix):
            params["topic"] = params["topic"][len(prefix):].strip()
            break
    
    # Handle "a/an" articles
    topic_words = params["topic"].split()
    if topic_words and topic_words[0].lower() in ["a", "an"]:
        params["topic"] = " ".join(topic_words[1:])
    
    # Handle "meme" keyword in various forms
    meme_keywords = ["meme about", "meme for", "meme of", "meme with", "meme"]
    topic_lower = params["topic"].lower()
    for keyword in meme_keywords:
        if keyword in topic_lower:
            # Extract text after the meme keyword
            parts = params["topic"].lower().split(keyword, 1)
            if len(parts) > 1 and parts[1].strip():
                params["topic"] = parts[1].strip()
                break
            elif keyword == "meme" and parts[0].strip():
                # If "meme" is at the end, use the part before it
                params["topic"] = parts[0].strip()
                break
    
    return params 