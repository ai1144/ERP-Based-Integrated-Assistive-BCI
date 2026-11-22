'''
Here I used this to understand how post requests work: https://www.w3schools.com/python/ref_requests_post.asp
This is what I used to understand status codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status 
I also used this for the Gradio Interface: https://www.gradio.app/guides/blocks-and-event-listeners 
And this is the theme I used for the interface: https://huggingface.co/spaces/freddyaboulton/dracula_revamped 
I used ChatGPT to help me with the dynamic button change (in def update_ui) because I didn't find "gr.update()" in the Gradio Quick Start guide
'''

import gradio as gr
import os
from project_config import BACKEND_URL, DATA_DIR
import random
import requests
from ultimate_image_classification import classify_image
from ultimate_wav2vec import AudioClassifier

def random_file_selector(base_path): # function to select one random file
    categories = [] 
    for d in os.listdir(base_path): # loop files in directory
        if os.path.isdir(os.path.join(base_path, d)): # checking if dir exists
            categories.append(d)
    selected_category = random.choice(categories)
    category_path = os.path.join(base_path, selected_category)
    image_files = []
    for f in os.listdir(category_path):
        if f.endswith(".jpeg"):
            image_files.append(os.path.join(category_path, f))
    audio_files = []
    for f in os.listdir(category_path):
        if f.endswith(".wav"):
            audio_files.append(os.path.join(category_path, f))
    if image_files and audio_files:
        selected_image = random.choice(image_files)
        selected_audio = random.choice(audio_files)
        return selected_image, selected_audio
    else:
        return None, None
'''
def process(image_path, audio_path, ngrok_url):
    full_url = f"{ngrok_url.rstrip('/')}/uploadfile/"
    image_category, image_score = classify_image(image_path)
    audio_classifier = AudioClassifier()
    reference_embeddings = audio_classifier.create_embeddings_for_folder(str(DATA_DIR / 'audios_categories'))
    similarities = audio_classifier.classify_audio(audio_path, reference_embeddings)
    best_match = max(similarities, key=similarities.get)
    data = {
        "aud": best_match,
        "img": image_category
    }
    with open(image_path, "rb") as file:
        files = {"file": (os.path.basename(image_path), file)}
        response = requests.post(full_url, files=files, data=data, timeout=120)
        if response.status_code == 200:
            model_response = response.json()["response"]
            output = f"Image category - Best match: {image_category} \nImage similarity: {image_score:.3f}\n" + model_response
            return output
        else:
            return f"Failed to upload file. Status code: {response.status_code}\n{response.content}"
'''
def process(image_path, audio_path, ngrok_url):
    full_url = f"{ngrok_url.rstrip('/')}/uploadfile/" # fiixing the ngrok url name to avoid error 404 file upload
    image_category, image_score = classify_image(image_path) # classify image
    audio_classifier = AudioClassifier() # calling other modules imported
    reference_embeddings = audio_classifier.create_embeddings_for_folder(str(DATA_DIR / 'audios_categories'))
    similarities = audio_classifier.classify_audio(audio_path, reference_embeddings)
    best_match = max(similarities, key=similarities.get)  # classify audio
    data = {
        "aud": best_match,
        "img": image_category
    } # data to send to qwen model
    with open(image_path, "rb") as file: # using binary mode to prevent encoding issues 
        files = {"file": (os.path.basename(image_path), file)}
        response = requests.post(full_url, files=files, data=data, timeout=120) # post request
        if response.status_code == 200:
            model_response = response.json()["response"]
            return model_response  
        else:
            return f"error w/ file upload, status code: {response.status_code}"

def update_ui(state_pressed, ngrok_url): 
    if not state_pressed:
        selected_image, selected_audio = random_file_selector(str(DATA_DIR / 'examples'))
        if selected_image and selected_audio:
            output_text = process(selected_image, selected_audio, ngrok_url)
            return ("P300 Detected", selected_image, selected_audio, output_text, True, gr.update(visible=True), gr.update(visible=True))
    return "P300 Component Detected", None, None, None, state_pressed, gr.update(), gr.update()

with gr.Blocks(theme="freddyaboulton/dracula_revamped") as app: # importing the theme
    gr.Markdown("# ERP Based Cognitive Interface Prototype") 
    ngrok_input = gr.Textbox(label="Model API URL", value=BACKEND_URL, placeholder="http://127.0.0.1:8000") 
    with gr.Row():
        button = gr.Button("No ERP Detected") # setting button up for change after interaction
    with gr.Row(visible=False) as content:
        with gr.Column(scale=1):
            gr.Markdown("## Visual Field")
            img = gr.Image()
        with gr.Column(scale=1):
            gr.Markdown("## Auditory Field")
            audio = gr.Audio()
    advice = gr.Textbox(label="Advice", visible=False)
    button.click(update_ui, inputs=[gr.State(value=False), ngrok_input], outputs=[button, img, audio, advice, gr.State(), content, advice])
if __name__ == "__main__":
    app.launch() 