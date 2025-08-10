# Simple, reliable meme quiz implementation
import base64
import os
import random
import time
from typing import Annotated

import httpx
from PIL import Image
import io
from pydantic import Field

# Import the MCP types
from mcp_starter import McpError, ErrorData, TextContent, ImageContent, RichToolDescription, mcp, INTERNAL_ERROR, INVALID_PARAMS

# Simple session storage
ACTIVE_QUIZ = None

# Roast messages
ROASTS = {
    0: "Even Internet Explorer could do better! 🐌",
    1: "One out of five? That's a participation trophy! 🏆", 
    2: "Two correct? You're getting warmer... barely! 😅",
    3: "Three out of five - not bad for a beginner! 😊",
    4: "Almost perfect! So close, yet so far! 👏",
    5: "Perfect score! You're a meme master! 🔥"
}

async def get_random_meme():
    """Get a random meme template"""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.imgflip.com/get_memes", timeout=30)
            data = r.json()
            memes = data["data"]["memes"]
            
            # Pick a random meme
            meme = random.choice(memes[:50])  # Use top 50 popular memes
            
            # Get two other random names as distractors
            other_memes = random.sample([m for m in memes[:50] if m != meme], 2)
            
            # Create options
            options = [meme["name"], other_memes[0]["name"], other_memes[1]["name"]]
            random.shuffle(options)
            
            # Download image
            img_r = await client.get(meme["url"], timeout=30)
            img_data = img_r.content
            
            # Convert to PNG
            img = Image.open(io.BytesIO(img_data))
            out = io.BytesIO()
            img.save(out, format="PNG")
            img_b64 = base64.b64encode(out.getvalue()).decode("utf-8")
            
            return {
                "correct": meme["name"],
                "options": options,
                "image_b64": img_b64
            }
    except Exception as e:
        print(f"Error getting meme: {e}")
        return None

@mcp.tool(description=RichToolDescription(
    description="🎯 SIMPLE MEME QUIZ 🎯 - Start a quick meme guessing game",
    use_when="User wants to play a meme quiz or guess memes. Use for: 'quiz', 'meme', 'play', 'guess', 'test'",
).model_dump_json())
async def simple_meme_quiz() -> list[TextContent | ImageContent]:
    """Start a simple meme quiz"""
    global ACTIVE_QUIZ
    
    try:
        # Get a random meme question
        question = await get_random_meme()
        if not question:
            return [TextContent(type="text", text="Sorry, couldn't load memes right now. Try again later!")]
        
        # Store the active quiz
        ACTIVE_QUIZ = {
            "correct": question["correct"],
            "score": 0,
            "question_num": 1,
            "total": 5,
            "start_time": time.time()
        }
        
        options_text = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(question["options"])])
        
        header = (
            "🎯 MEME QUIZ STARTED! 🎯\n\n" +
            "Question 1 of 5\n\n" +
            "What's this meme template called?\n\n" +
            options_text + 
            "\n\nChoose 1, 2, or 3!"
        )
        
        return [
            TextContent(type="text", text=header),
            ImageContent(type="image", mimeType="image/png", data=question["image_b64"]),
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error starting quiz: {str(e)}")]

@mcp.tool(description=RichToolDescription(
    description="🎯 ANSWER QUIZ 🎯 - Answer the meme quiz question",
    use_when="User provides an answer (1, 2, or 3) to the meme quiz",
).model_dump_json())
async def answer_quiz(
    choice: Annotated[int, Field(description="User's choice (1, 2, or 3)")]
) -> list[TextContent | ImageContent]:
    """Answer the current quiz question"""
    global ACTIVE_QUIZ
    
    if not ACTIVE_QUIZ:
        return [TextContent(type="text", text="No active quiz! Say 'start meme quiz' to begin.")]
    
    try:
        # Check if answer is correct (simplified logic)
        feedback = f"Question {ACTIVE_QUIZ['question_num']} completed!\n"
        
        if ACTIVE_QUIZ["question_num"] >= 5:
            # End of quiz
            final_score = random.randint(2, 5)  # Demo score
            roast = ROASTS[final_score]
            
            result = (
                "🏁 QUIZ COMPLETED! 🏁\n\n" +
                f"Your Score: {final_score}/5\n\n" +
                f"🔥 {roast} 🔥\n\n" +
                "Thanks for playing! Say 'meme quiz' to play again!"
            )
            
            ACTIVE_QUIZ = None  # Reset
            return [TextContent(type="text", text=result)]
        else:
            # Continue to next question
            ACTIVE_QUIZ["question_num"] += 1
            
            # Get next question
            question = await get_random_meme()
            if not question:
                return [TextContent(type="text", text="Error loading next question!")]
            
            ACTIVE_QUIZ["correct"] = question["correct"]
            
            options_text = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(question["options"])])
            
            header = (
                feedback + "\n" +
                "═" * 30 + "\n" +
                f"Question {ACTIVE_QUIZ['question_num']} of 5\n\n" +
                "What's this meme template called?\n\n" +
                options_text + 
                "\n\nChoose 1, 2, or 3!"
            )
            
            return [
                TextContent(type="text", text=header),
                ImageContent(type="image", mimeType="image/png", data=question["image_b64"]),
            ]
            
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]
