# Clean implementation of the meme quiz tools
import base64
import hashlib
import hmac
import io
import os
import random
import time
from typing import Annotated

import httpx
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

# Import the MCP types - adjust these imports based on your actual setup
from mcp_starter import McpError, ErrorData, TextContent, ImageContent, RichToolDescription, mcp, INTERNAL_ERROR, INVALID_PARAMS

# Constants
TOKEN = os.getenv("QUIZ_SECRET", "default_secret_key")

# Quiz Sessions and State
QUIZ_SESSIONS: dict[str, dict] = {}
USED_MEMES: set[str] = set()

# Roast messages based on score ranges
ROAST_MESSAGES = {
    0: [
        "Even a potato could do better! 🥔",
        "Did you even try? My grandma knows more memes!",
        "Error 404: Meme knowledge not found",
    ],
    1: [
        "One right answer? What are you, using Internet Explorer? 🐌",
        "Congratulations on the participation trophy! 🏆",
        "You're about as good at this as a cat is at swimming",
    ],
    2: [
        "Two correct? You're like Internet Explorer - not completely useless, but close",
        "Half wrong is still... pretty wrong 😅",
        "You're the human equivalent of a loading screen",
    ],
    3: [
        "Three out of five? Not bad... for a boomer 👴",
        "You're like WiFi on an airplane - technically working, but barely",
        "Mediocrity at its finest! 🌟",
    ],
    4: [
        "Almost perfect! But almost doesn't get you internet points 😏",
        "So close, yet so far! Like my relationship with productivity",
        "Four out of five? That's like a pizza without cheese - good but not great",
    ],
    5: [
        "Perfect score! But can you do it again? (Probably not) 😎",
        "Okay meme lord, we get it, you have no life outside Reddit",
        "5/5! Your parents must be so proud of this achievement 🏆",
    ]
}

def _make_session_id(secret: str) -> str:
    """Generate a simple session ID for quiz tracking"""
    nonce = os.urandom(8).hex()
    ts = str(int(time.time()))
    return f"quiz_{nonce}_{ts}"

def _cleanup_expired_sessions():
    """Remove expired quiz sessions and reset used memes tracking after 1 hour"""
    global USED_MEMES
    current_time = time.time()
    
    # Reset used memes roughly every hour
    if current_time % 3600 < 60:
        USED_MEMES = set()
    
    # Remove expired quiz sessions (older than 30 minutes)
    expired = []
    for sid in QUIZ_SESSIONS:
        if current_time - QUIZ_SESSIONS[sid]["created_at"] > 1800:  # 30 min timeout
            expired.append(sid)
    
    for sid in expired:
        print(f"Removing expired session: {sid}")
        del QUIZ_SESSIONS[sid]
    
    if expired:
        print(f"Cleaned up {len(expired)} expired sessions")

def _reset_user_sessions():
    """Reset all active quiz sessions"""
    global QUIZ_SESSIONS
    QUIZ_SESSIONS = {}

async def _download_image(client: httpx.AsyncClient, url: str) -> bytes:
    """Download and validate an image from a URL"""
    try:
        # Add user agent to avoid potential blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Download image
        r = await client.get(url, headers=headers, timeout=30, follow_redirects=True)
        r.raise_for_status()
        
        # Check content type
        content_type = r.headers.get("content-type", "").lower()
        if not any(mime in content_type for mime in ["image/jpeg", "image/png", "image/gif"]):
            raise ValueError(f"Invalid content type: {content_type}")
        
        # Get image data
        img_data = r.content
        
        # Verify it's a valid image
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to PNG
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
        
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        raise

async def _fetch_imgflip_templates(client: httpx.AsyncClient) -> list[dict]:
    """Fetch meme templates from imgflip API"""
    try:
        print("Fetching templates from imgflip...")
        r = await client.get("https://api.imgflip.com/get_memes", timeout=30)
        r.raise_for_status()
        
        payload = r.json()
        memes = (payload.get("data") or {}).get("memes") or []
        print(f"Found {len(memes)} templates from imgflip")
        
        results = []
        for m in memes:
            name = m.get("name", "")
            url = m.get("url")
            if name and url:
                results.append({
                    "id": str(m.get("id")),
                    "name": name,
                    "image_url": url,
                })
        
        print(f"Filtered to {len(results)} valid templates")
        return results
    except Exception as e:
        print(f"Error fetching imgflip templates: {str(e)}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch templates: {str(e)}"))

async def _build_quiz_questions(client: httpx.AsyncClient, source: str, total_questions: int) -> list[dict]:
    global USED_MEMES
    candidates = await _fetch_imgflip_templates(client)

    # Unique templates with valid name and image, prioritizing unused memes
    candidates = [c for c in candidates if (c.get("name") and c.get("image_url"))]
    unused_candidates = [c for c in candidates if c["name"] not in USED_MEMES]
    
    # If we don't have enough unused memes, mix in some used ones
    if len(unused_candidates) < max(6, total_questions + 2):
        random.shuffle(candidates)
        candidates = unused_candidates + [c for c in candidates if c not in unused_candidates]
    else:
        candidates = unused_candidates
        random.shuffle(candidates)
    
    if len(candidates) < max(6, total_questions + 2):
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Not enough templates to create a full quiz."))
    
    # Print for debugging
    print(f"Selected candidates: {[c['name'] for c in candidates[:total_questions]]}")

    selected = candidates[: total_questions]
    questions: list[dict] = []
    for correct_tpl in selected:
        try:
            # Track this meme as used
            USED_MEMES.add(correct_tpl["name"])
            
            # Pick exactly 2 distractors to make 3 total options
            distractor_pool = [c for c in candidates if c["name"] != correct_tpl["name"]]
            if len(distractor_pool) < 2:
                print(f"Not enough distractors for {correct_tpl['name']}")
                continue
            
            distractors = random.sample(distractor_pool, 2)
            options = [correct_tpl["name"], distractors[0]["name"], distractors[1]["name"]]
            random.shuffle(options)
            
            # Ensure we have exactly 3 options
            if len(options) != 3:
                print(f"Invalid number of options for {correct_tpl['name']}: {len(options)}")
                continue

            # Download and process image
            print(f"Downloading image for {correct_tpl['name']} from {correct_tpl['image_url']}")
            img_bytes = await _download_image(client, correct_tpl["image_url"])
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            print(f"Successfully processed image for {correct_tpl['name']}")
            
            questions.append({
                "correct": correct_tpl["name"],
                "options": options,
                "image_b64": img_b64,
            })
        except Exception as e:
            print(f"Error processing question for {correct_tpl['name']}: {str(e)}")
            # Skip this template and try the next one
            continue
    
    # Ensure we have exactly 5 questions
    if len(questions) < 5:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Could only create {len(questions)} questions, need 5"))
    
    return questions[:5]  # Take exactly 5 questions

async def _get_roast_meme(client: httpx.AsyncClient, score: int) -> tuple[str, bytes]:
    """Get a meme template suitable for roasting based on score"""
    # Define meme templates good for roasting
    roast_templates = [
        "Condescending-Wonka", "Evil-Kermit", "Disaster-Girl", "Spongebob-Mock",
        "Roll-Safe", "Waiting-Skeleton", "Laughing-Men-In-Suits", "Ancient-Aliens"
    ]
    
    # Try to get a roast meme from imgflip
    r = await client.get("https://api.imgflip.com/get_memes", timeout=30)
    if r.status_code == 200:
        data = r.json()
        memes = data.get("data", {}).get("memes", [])
        roast_memes = [m for m in memes if any(rt.lower() in m.get("name", "").lower() for rt in roast_templates)]
        if roast_memes:
            meme = random.choice(roast_memes)
            img_url = meme.get("url")
            if img_url:
                r = await client.get(img_url, timeout=30)
                if r.status_code == 200:
                    return meme["name"], r.content
    
    # Fallback to a basic meme if imgflip fails
    return "Generic Roast", b""

@mcp.tool(description=RichToolDescription(
    description="🎯 MEME QUIZ GAME 🎯 - Interactive meme quiz with 5 questions and scoring",
    use_when="User wants to play a meme quiz or says 'ready' to start. For quiz requests: 'play', 'quiz', 'meme', 'guess', 'test', 'let's play', 'meme quiz', 'quiz me'. For starting: 'ready', 'start', 'begin', 'go'.",
    side_effects="Shows quiz intro first, then starts actual quiz when user says ready.",
).model_dump_json())
async def meme_quiz_game(
    action: Annotated[str, Field(description="Action: 'intro' for first time, 'start' if user said ready/start")] = "intro",
    source: Annotated[str, Field(description="Template source: 'imgflip' or 'memegen'")] = "imgflip",
) -> list[TextContent | ImageContent]:
    try:
        # Clean up any expired sessions
        _cleanup_expired_sessions()
        
        if action == "start":
            # Start the actual quiz
            total_questions = 5
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
            
            print(f"Created new quiz session: {session_id}")
            print(f"Session contains {len(questions)} questions")

            q = questions[0]
            if not q.get("image_b64"):
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="No image data found"))
            
            base_img_bytes = base64.b64decode(q["image_b64"])
            QUIZ_SESSIONS[session_id]["question_start_ts"] = time.time()

            options_text = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(q["options"])])
            header = (
                "🎯 QUIZ STARTED! 🎯\n\n" +
                "═══════════════════════════════════════\n" +
                f"QUESTION 1 of {total_questions}\n" +
                "═══════════════════════════════════════\n\n" +
                "What's this meme template called?\n\n" +
                options_text + 
                "\n\n⏰ You have 20 seconds! Choose 1, 2, or 3:\n" +
                f"📝 Session ID: {session_id}"
            )
            return [
                TextContent(type="text", text=header),
                ImageContent(type="image", mimeType="image/png", data=base64.b64encode(base_img_bytes).decode("utf-8")),
            ]
        
        elif action == "reset":
            _reset_user_sessions()
            return [TextContent(type="text", text="🔄 Quiz reset! Say 'let's play guess the meme' to start fresh!")]
        
        else:
            # Show introduction and instructions (default intro behavior)
            intro_text = (
                "🎯 WELCOME TO THE ULTIMATE MEME TEMPLATE QUIZ CHALLENGE! 🎯\n\n" +
                "📋 QUIZ RULES:\n" +
                "• This quiz consists of 5 exciting questions\n" +
                "• Each question shows a meme template image\n" +
                "• You'll get 3 options to choose from for each question\n" +
                "• You have 20 seconds to answer each question\n" +
                "• Score points for correct answers\n" +
                "• Get a final score and custom roast meme at the end!\n\n" +
                "🎮 HOW TO PLAY:\n" +
                "• Look at the meme image carefully\n" +
                "• Choose the correct template name (1, 2, or 3)\n" +
                "• Type your answer quickly - time is ticking!\n" +
                "• Watch your score grow with each correct answer\n\n" +
                "🏆 SCORING SYSTEM:\n" +
                "• 5/5 = MEME MASTER 🔥\n" +
                "• 4/5 = MEME EXPERT 👏\n" +
                "• 3/5 = MEME ENTHUSIAST 😊\n" +
                "• 2/5 = MEME NOVICE 📚\n" +
                "• 1/5 = MEME BEGINNER 🤔\n" +
                "• 0/5 = MEME NEWBIE 😅\n\n" +
                "🚀 Ready to test your meme knowledge?\n" +
                "💡 Just say 'I'm ready to start the quiz!' to begin!"
            )
            return [TextContent(type="text", text=intro_text)]
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to start quiz: {str(e)}"))

@mcp.tool(description=RichToolDescription(
    description="🎯 ANSWER MEME QUIZ 🎯 - Submit answer for the meme quiz game",
    use_when="User provides an answer (1, 2, or 3) during the meme quiz game. Use this to process answers and advance to next question or show final results.",
).model_dump_json())
async def answer_meme_quiz_game(
    session_id: Annotated[str, Field(description="The session_id returned by meme_quiz_game")],
    choice_index: Annotated[int | None, Field(description="Your selected option index (1-3)")] = None,
    answer_text: Annotated[str | None, Field(description="Alternatively provide the template name directly")] = None,
    command: Annotated[str | None, Field(description="Optional command: 'skip', 'quit', 'help'")] = None,
) -> list[TextContent | ImageContent]:
    try:
        # Handle commands first
        if command:
            command = command.lower().strip()
            if command == "help":
                return [TextContent(type="text", text=(
                    "🎮 Meme Quiz Commands:\n\n"
                    "- Enter 1-3 to choose an answer\n"
                    "- Type 'skip' to skip current question\n"
                    "- Type 'quit' to end the quiz\n"
                    "- Type 'help' to see this message\n"
                    "- Type 'reset' or 'restart' to start over"
                ))]
            elif command == "quit":
                if session_id in QUIZ_SESSIONS:
                    del QUIZ_SESSIONS[session_id]
                return [TextContent(type="text", text="Quiz ended. Type 'start' to begin a new quiz!")]
            elif command == "skip":
                choice_index = 0  # Will count as wrong but move to next question
        
        # Check if session exists
        if session_id not in QUIZ_SESSIONS:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Session {session_id} not found. Please start a new quiz with 'let's play guess the meme'"))
        
        state = QUIZ_SESSIONS[session_id]
        
        # Add debug info
        print(f"Processing answer for session {session_id}")
        print(f"Current sessions: {list(QUIZ_SESSIONS.keys())}")

        questions: list[dict] = state["questions"]
        idx: int = state["index"]
        score: int = state["score"]
        total = len(questions)
        
        if choice_index is None and not answer_text:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide choice_index (1-3) or answer_text."))

        # Get current question
        q = questions[idx]
        options = q["options"]
        correct_name = q["correct"]
        
        # Format user's answer
        if answer_text:
            given = answer_text.strip()
            user_answer = given
        else:
            given = str(choice_index)
            user_answer = options[choice_index - 1] if 1 <= choice_index <= len(options) else "Invalid choice"

        # Check if answer is correct
        is_correct = False
        if answer_text:
            is_correct = given.lower() == correct_name.lower()
        elif choice_index and 1 <= choice_index <= len(options):
            is_correct = options[choice_index - 1] == correct_name
        
        # Update score and show feedback
        if is_correct:
            state["score"] += 1
            feedback = "🎉 CORRECT! Awesome! +1 point! "
        else:
            feedback = "❌ WRONG! No points this time. "
        
        # Show answer feedback with current score
        current_score = state["score"]
        feedback += f"The correct answer was: {correct_name}\n\n"
        feedback += f"📊 CURRENT SCORE: {current_score}/{idx + 1} questions answered\n"
        
        # If this was the last question, show final score and roast
        if idx + 1 >= total:
            score = state["score"]
            percentage = (score / total) * 100
            roast = random.choice(ROAST_MESSAGES[score])
            
            # Get a roast meme
            async with httpx.AsyncClient() as client:
                meme_name, meme_bytes = await _get_roast_meme(client, score)
            
            feedback += "\n\n🏁 QUIZ COMPLETED! CONGRATULATIONS! 🏁\n"
            feedback += "═══════════════════════════════════════\n"
            feedback += "           📊 FINAL RESULTS 📊\n"
            feedback += "═══════════════════════════════════════\n"
            feedback += f"🎯 YOUR FINAL SCORE: {score} out of {total} ({percentage:.0f}%)\n"
            feedback += f"🏆 PERFORMANCE LEVEL: "
            if score == 5:
                feedback += "MEME MASTER! 🔥 Perfect score!\n"
            elif score == 4:
                feedback += "MEME EXPERT! 👏 Almost perfect!\n"
            elif score == 3:
                feedback += "MEME ENTHUSIAST! 😊 Not bad!\n"
            elif score == 2:
                feedback += "MEME NOVICE! 📚 Keep practicing!\n"
            elif score == 1:
                feedback += "MEME BEGINNER! 🤔 You tried!\n"
            else:
                feedback += "MEME NEWBIE! 😅 Better luck next time!\n"
            
            feedback += "\n" + "═" * 39 + "\n"
            feedback += f"🔥 ROAST TIME! 🔥\n"
            feedback += f"{roast}\n\n"
            feedback += "Here's a meme that perfectly represents your performance...\n"
            feedback += "You deserve this! 😂\n\n"
            feedback += "🎮 Want to play again? Just say 'let's play guess the meme'!"
            
            # Clean up the session
            del QUIZ_SESSIONS[session_id]
            
            return [
                TextContent(type="text", text=feedback),
                ImageContent(type="image", mimeType="image/png", data=base64.b64encode(meme_bytes).decode("utf-8")),
            ]
        else:
            # Move to next question
            state["index"] += 1
            next_q = questions[state["index"]]
            
            # Show feedback and next question
            feedback += f"\n⏭️ Moving to next question...\n\n"
            feedback += "═══════════════════════════════════════\n"
            feedback += f"🎯 QUESTION {state['index'] + 1} of {total} 🎯\n"
            feedback += "═══════════════════════════════════════\n\n"
            feedback += "What's this meme template called?\n\n"
            options_text_next = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(next_q["options"])])
            feedback += options_text_next
            feedback += "\n\n⏰ You have 20 seconds! Choose 1, 2, or 3:"
            feedback += f"\n💡 Current streak: {current_score} correct answers so far!"
            
            # Start timer for next question
            state["question_start_ts"] = time.time()
            
            return [
                TextContent(type="text", text=feedback),
                ImageContent(type="image", mimeType="image/png", data=base64.b64encode(base64.b64decode(next_q["image_b64"]))),
            ]
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to evaluate quiz answer: {str(e)}"))
