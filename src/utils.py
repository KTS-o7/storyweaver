import os
import asyncio
import streamlit as st
from mirascope.core import groq, Messages
from mirascope.core.groq import GroqCallParams
from pypollinations import TextClient, ImageClient, TextGenerationRequest, ImageGenerationRequest
from .models import StoryBoard, StoryTopic, StoryText, StoryPartPrompts, StroyImagePrompts
from pypollinations.models.base import ImageModel
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import random


groq_model = "llama-3.3-70b-versatile"
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = groq_api_key
groq_call_params = GroqCallParams(temperature=0.8, max_tokens=16384)

@groq.call(model=groq_model, call_params=groq_call_params, response_model=StoryPartPrompts, json_mode=True)
async def generate_story_part_prompts_from_prompt(prompt: str) -> str:
    return [Messages.System("""
                        You are a children's Storyteller. You are amazing at using your imagination to create stories that are both entertaining and educational. Your audience are children of age 3-10.
                        - You will NOT use any copyrighted material.
                        - You will NOT use any inappropriate content.
                        - You will NOT use any content that is not suitable for children.
                        - You will be given a prompt.
                        - Based on the prompt you will create multiple prompts for each part of the story.
                        - Each story must have atleast 4 paragraphs.
                        - Later on these prompts will be used by LLMs to generate the story.
                        """),
            Messages.User(f"""
                    You are given the following prompt:
                    ```{prompt}```
                    - Generate prompts for each part of the story.
                    - Output will be a JSON object of list of strings, where each string is a prompt for each part of the story.
                    """)]

@groq.call(model=groq_model, call_params=groq_call_params, response_model=StroyImagePrompts, json_mode=True)
async def generate_story_image_prompts(story_part_prompts: StoryPartPrompts) -> str:
    return [Messages.System("""
                    You are an expert in image design prompts. You are amazing at using your imagination to create images that are both entertaining and educational. Your audience are children of age 3-10.
                        - You will NOT use any inappropriate content.
                        - You will NOT use any content that is not suitable for children.
                        - You will be given a list of prompts for each part of the story.
                        - Based on the prompts you will Create one image prompt for each of the story points.
                        - Later on these prompts will be used by Pollinations to generate the images.
                        - Keep prompts simple and amazingly creative.
                    """),
            Messages.User(f"""
                        You are given the following prompts for each part of the story:
                        ```{story_part_prompts}```
                        - Generate one image prompt for each of the story points.
                        - Output will be a JSON object of list of strings, where each string is an image prompt for each of the story points.
                    """)]

class StoryGenerator:
    def __init__(self, story_topic: StoryTopic):
        self.text_model = story_topic.text_model
        self.image_model = story_topic.image_model
        self.text_client = TextClient()
        self.image_client = ImageClient()

    async def generate_story_part_prompts_from_prompt(self, prompt: str) -> StoryPartPrompts:
        prompts = await generate_story_part_prompts_from_prompt(prompt)
        return prompts

    async def generate_story_image_prompts(self, story_part_prompts: StoryPartPrompts) -> StroyImagePrompts:
        prompts = await generate_story_image_prompts(story_part_prompts)     
        return prompts

    async def generate_story_text(self, story_part_prompts: StoryPartPrompts, timeout: int = 30) -> StoryText:
        story_parts = []
        story_part_prompts = story_part_prompts.story_part_prompts
        
        async def process_prompt(prompt, previous_parts):
            try:
                context = ""
                if previous_parts:
                    context = "Previous parts of the story:\n" + "\n".join(previous_parts) + "\nContinue the story with:"
                
                full_prompt = f"{context}\n{prompt}" if context else prompt
                message = {'role': 'user', 'content': full_prompt}
                messages = [message]
                input = TextGenerationRequest(messages=messages, model=self.text_model, temperature=0.8)
                
                response = await asyncio.wait_for(
                    self.text_client.generate(input),
                    timeout=timeout
                )
                return response.content
            except TimeoutError:
                print(f"Text generation timed out after {timeout} seconds")
                return "Story generation timed out. Please try again."
            except Exception as e:
                print(f"Error generating text: {str(e)}")
                return "An error occurred during story generation."

        # Process prompts sequentially to maintain story coherence
        for prompt in story_part_prompts:
            part = await process_prompt(prompt, story_parts)
            if isinstance(part, str) and not part.startswith("Error") and not part.startswith("Story generation timed out"):
                story_parts.append(part)

        return StoryText(title="Story", story_points=story_parts)

    async def generate_story_images(self, story_image_prompts: StroyImagePrompts, timeout: int = 90) -> list[bytes]:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        async def process_image(prompt: str) -> Optional[bytes]:
            try:
                models = [ImageModel.FLUX, ImageModel.FLUX_3D, ImageModel.FLUX_PRO, ImageModel.TURBO]
                input = ImageGenerationRequest(
                    model=models[random.randint(0, len(models) - 1)], 
                    prompt=prompt,
                    safe=False,
                    nologo=True,
                    private=True,
                    enhance=True,
                    width=512,
                    height=512
                )
                # Wrap the API call with timeout
                response = await asyncio.wait_for(
                    self.image_client.generate(input),
                    timeout=timeout
                )
                return response.image_bytes
            except asyncio.TimeoutError:
                print(f"Image generation timed out after {timeout} seconds for prompt: {prompt[:100]}...")
                return None
            except Exception as e:
                if hasattr(e, 'status'):
                    print(f"HTTP {e.status} error for prompt: {prompt[:100]}...")
                    print(f"Response: {str(e)}")
                else:
                    print(f"Error generating image: {type(e).__name__}: {str(e)}")
                    print(f"Prompt: {prompt[:100]}...")
                return None

        print(f"Starting generation of {len(story_image_prompts.image_prompts)} images...")
        tasks = [process_image(prompt) for prompt in story_image_prompts.image_prompts]
        images = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful generations
        successful = sum(1 for img in images if img is not None)
        print(f"Successfully generated {successful}/{len(images)} images")
        
        # Filter out None values from failed generations
        return [img for img in images if img is not None]

    def generate_story_board(self, story_text: StoryText, story_images: list[bytes]) -> StoryBoard:
        return StoryBoard(story_text=story_text, story_images=story_images)

    async def generate_story(self, prompt: str) -> StoryBoard:
        try:
            story_part_prompts = await self.generate_story_part_prompts_from_prompt(prompt)
            story_image_prompts = await self.generate_story_image_prompts(story_part_prompts)
            
            # Run text and image generation concurrently
            story_text, story_images = await asyncio.gather(
                self.generate_story_text(story_part_prompts),
                self.generate_story_images(story_image_prompts)
            )
            
            return self.generate_story_board(story_text, story_images)
        except Exception as e:
            print(f"Error generating story: {str(e)}")
            raise

    async def list_text_models(self) -> list[str]:
        return await self.text_client.list_models()

    async def list_image_models(self) -> list[str]:
        return await self.image_client.list_models()