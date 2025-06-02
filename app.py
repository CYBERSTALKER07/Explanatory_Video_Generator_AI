import asyncio
import os
import shutil
import subprocess
import tempfile
from typing import List
import logging
import re

import streamlit as st
from moviepy.editor import concatenate_videoclips, VideoFileClip
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()
api_key = os.getenv("AIzaSyAdYhRe7Rm9si3SV3RtCpsjRORwYZnoSZk")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Gemini LLM setup
gemini_llm = GeminiModel(
    "gemini-2.0-flash", provider=GoogleGLAProvider(api_key=api_key)
)

# Models
class ChapterDescription(BaseModel):
    title: str = Field(description="Title of the chapter.")
    explanation: str = Field(
        description="Detailed explanation of the chapter’s content, including how Manim should visualize it."
    )

class VideoOutline(BaseModel):
    title: str
    chapters: List[ChapterDescription]

class ManimCode(BaseModel):
    code: str = Field(description="Complete, runnable Manim code for a single scene.")

# Agents
outline_agent = Agent(
    model=gemini_llm,
    result_type=VideoOutline,
    system_prompt="""
    You are a video script writer. Write a title and up to 3 chapters with detailed visual instructions for Manim.
    Include LaTeX, animation guidance, transitions, and layout hints. Do not include code.
    """
)

manim_agent = Agent(
    model=gemini_llm,
    result_type=ManimCode,
    system_prompt="""
    You are a Manim code generator. Create runnable Manim code for one chapter.
    Include imports, one scene, valid Python comments only, and follow the explanation closely.
    """
)

code_fixer_agent = Agent(
    model=gemini_llm,
    result_type=ManimCode,
    system_prompt="""
    You are a Manim code debugger. Fix code based on the provided error.
    Ensure imports, correct scene, valid Python comments, and runnable output.
    """
)

# Functions
def generate_manim_code(chapter: ChapterDescription) -> str:
    logging.info(f"Generating Manim code for: {chapter.title}")
    result = manim_agent.run_sync(f"title: {chapter.title}. Explanation: {chapter.explanation}")
    return result.data.code

def fix_manim_code(error: str, code: str) -> str:
    logging.info("Fixing Manim code after error...")
    result = code_fixer_agent.run_sync(f"Error: {error}\nCurrent Code:\n{code}")
    return result.data.code

def generate_video_outline(concept: str) -> VideoOutline:
    logging.info(f"Generating outline for: {concept}")
    result = outline_agent.run_sync(concept)
    return result.data

def extract_class_name(code: str) -> str:
    match = re.search(r"class\s+(\w+)\s*\(\s*\w*Scene\s*\):", code)
    if match:
        return match.group(1)
    else:
        raise ValueError("Could not extract class name from Manim code.")

def create_video_from_code(code: str, chapter_num: int) -> str:
    """Runs Manim and returns path to the generated video."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
        temp_file.write(code)
        temp_file_name = temp_file.name

    try:
        command = ["manim", temp_file_name, "-ql", "--disable_caching"]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=60)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=command,
                output=stdout,
                stderr=stderr
            )

    except subprocess.TimeoutExpired:
        logging.error(f"Manim process timed out for chapter {chapter_num}.")
        process.kill()
        raise
    except FileNotFoundError:
        raise FileNotFoundError("Manim is not installed or not in PATH.")
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

    class_name = extract_class_name(code)
    video_path = os.path.join("./media/videos/temp/480p15/", f"{class_name}.mp4")
    return video_path

async def generate_video(concept: str):
    outline = generate_video_outline(concept)
    video_files = []

    for i, chapter in enumerate(outline.chapters):
        manim_code = generate_manim_code(chapter)
        attempts = 0
        success = False

        while attempts < 2 and not success:
            try:
                video_file = create_video_from_code(manim_code, i + 1)
                if os.path.exists(video_file):
                    video_files.append(video_file)
                    success = True
                else:
                    raise FileNotFoundError("Video file not created.")
            except Exception as e:
                logging.error(f"Attempt {attempts + 1} failed: {e}")
                if attempts == 0:
                    manim_code = fix_manim_code(str(e), manim_code)
            attempts += 1

        if not success:
            logging.warning(f"Skipping chapter {i + 1}: {chapter.title}")
            st.warning(f"Failed to generate chapter {i + 1}: {chapter.title}")

    if video_files:
        try:
            clips = [VideoFileClip(vf) for vf in video_files if os.path.exists(vf)]
            final_path = "final_video.mp4"
            if clips:
                final = concatenate_videoclips(clips)
                final.write_videofile(final_path, codec="libx264", audio_codec="aac")
                final.close()
                for clip in clips:
                    clip.close()
                st.success("Video generation complete!")
                return final_path
            else:
                st.warning("No clips to combine.")
        except Exception as e:
            logging.error(f"Video combining error: {e}")
            st.error(f"Failed to combine video: {e}")
    else:
        st.warning("No video chapters were successfully generated.")
    return None

def main():
    st.title("Explanatory Video Generator")
    concept = st.text_input("Enter a concept:")

    if st.button("Generate Video"):
        if concept:
            with st.spinner("Generating video... Please wait."):
                final_video = asyncio.run(generate_video(concept))
            if final_video and os.path.exists(final_video):
                st.video(final_video)
            else:
                st.info("Video generation completed but no output video found.")
        else:
            st.warning("Please enter a concept.")

if __name__ == "__main__":
    main()
