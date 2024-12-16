import streamlit as st
import asyncio
from src.utils import StoryGenerator
from src.models import StoryTopic

st.set_page_config(page_title="AI Children's Story Generator", 
                   page_icon="🐉", 
                   layout="wide")

async def main():
    st.title("AI Children's Story Generator")
    
    with st.form("story_form"):
        prompt = st.text_area("Enter your story topic:", 
                           placeholder="Example: A friendly dragon learning to fly")
        
        text_model = st.selectbox("Select Text Model", 
                                    ["openai", "mistral","searchgpt","midijourney"],
                                    index=0)
        
        submit = st.form_submit_button("Generate Story")
        
    if submit and prompt:
        with st.spinner("Generating your story..."):
            story_topic = StoryTopic(
                prompt=prompt,
                text_model=text_model,
                image_model="flux"
            )
            generator = StoryGenerator(story_topic)
            
            story_board = await generator.generate_story(prompt)
            
            st.header(story_board.story_text.title)
            
            for text, image in zip(story_board.story_text.story_points, 
                                 story_board.story_images):
                cols = st.columns(2)
                with cols[0]:
                    st.write(text)
                with cols[1]:
                    st.image(image)

if __name__ == "__main__":
    asyncio.run(main())