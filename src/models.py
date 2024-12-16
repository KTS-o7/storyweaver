from pydantic import BaseModel,Field
from typing import Optional

class StoryTopic(BaseModel):
    prompt: str = Field(title="Topic", description="The topic of the story")
    text_model : Optional[str] = Field(title="Text Model", description="The LLM text model to use for the story")
    image_model : Optional[str] = Field(title="Image Model", description="The Pollinations image model to use for the story")
    
class StoryPartPrompts(BaseModel):
    story_part_prompts: list[str] = Field(title="Story Part Prompts", description="The prompts for each part of the story")
    
class StroyImagePrompts(BaseModel):
    image_prompts: list[str] = Field(title="Image Prompts", description="The image prompts for each of the story points")
       
class StoryText(BaseModel):
    title: str = Field(title="Title", description="The title of the story")
    story_points: list[str] = Field(title="Story Points", description="The parts of the story")
    
class StoryBoard(BaseModel):
    story_text: StoryText = Field(title="Story Text", description="The text of the story")
    story_images: list[bytes] = Field(title="Story Images", description="The images of the story")
    
class Message(BaseModel):
    role: str
    content: str