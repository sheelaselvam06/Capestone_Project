from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import faiss
import os, getpass 

os.environ['HUGGINGFACEHUB_API_TOKEN'] = getpass.getpass('Hugging Token: ')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1][:,0,:]
    embeddings = torch.nn.functional.normalize(embedding)
    return embeddings 

documents = [
    "A christmas carol is a novella by Charles Dicken, first published in 1843.",
    "It tells the story of sour and stingy ebenezer Scrooge's ideological, ethical, and emotional transformation after being visited by the ghost of Christmas Past, Present, and Yet to Come.",
    "The supernatural visits of jacob marley and the Ghost of Christmas Past, Present, and Yet to Come.",
    "The novella met with instant success and critical acclaim. It is regarded as one of the greatest christmas stories ever written."] 
chunk_emdeddings=get_embeddings(chunk)
index = faiss.IndexFlat(chunk_embeddings.shape[1])
index.add(chunk_embeddings.detch().numpy )