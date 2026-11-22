'''
Most of the code was adapted from the HuggingFace overview of the Wav2Vec model: https://huggingface.co/docs/transformers/en/model_doc/wav2vec2 
The parameters were also taken from that repository, but the code was slightly adapted to be object oriented with classes for readability and to later import it as a module in the ultimate_connect script
This was used as a guide for the torchaudio builtin functions: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
'''
import glob
import torchaudio
import torch
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AudioClassifier:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) 
        self.model = Wav2Vec2Model.from_pretrained(model_name) # this is from the huggingface overview, loading the pretrained model

    def preprocess_audio(self, file_path, target_sample_rate=16000): # preprocessing the .wav audio file
        audio, sample_rate = torchaudio.load(file_path, format="wav")
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sample_rate != target_sample_rate:
            resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            audio = resample_transform(audio)
        return audio.squeeze() # this overall function was made using the pytorch guide
    def get_audio_embedding(self, file_path): # embedding the audios
        audio = self.preprocess_audio(file_path)
        input_values = self.processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad(): # I adapted this part from the image classification script
            features = self.model(input_values).last_hidden_state
        return features.mean(dim=1) # only one vector
    def create_embeddings_for_folder(self, folder_path):
        embeddings = {}
        for file_path in glob.glob(f"{folder_path}/*.wav"):
            file_name = os.path.basename(file_path).replace(".wav", "") # renaming
            embedding = self.get_audio_embedding(file_path)
            embeddings[file_name] = embedding
        return embeddings
    def classify_audio(self, user_audio_path, reference_embeddings):
        user_embedding = self.get_audio_embedding(user_audio_path)
        similarities = {}
        for label, embedding in reference_embeddings.items():
            if user_embedding.dim() == 1:
                user_embedding = user_embedding.unsqueeze(0)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0) # from the pytorch guide as well
            similarity = torch.nn.functional.cosine_similarity(user_embedding, embedding, dim=1)
            similarities[label] = similarity.item()
        return similarities
