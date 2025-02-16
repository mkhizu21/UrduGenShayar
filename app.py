import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import re

# Load dataset (make sure the file exists in the same directory)
df = pd.read_csv("Cleaned_Roman_Urdu_Poetry.csv")

# Character Tokenizer
def char_tokenizer(text):
    vocab = sorted(set(text))
    token_to_id = {char: idx + 1 for idx, char in enumerate(vocab)}  # Reserve 0 for padding
    token_to_id['<PAD>'] = 0
    return token_to_id

# Tokenize the entire dataset
all_text = ' '.join(df['Poetry'].dropna().tolist())
token_to_id = char_tokenizer(all_text)
vocab_size = len(token_to_id)

# LSTM Model
class ShayariLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2):
        super(ShayariLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size, hidden_dim, num_layers, device):
        return (torch.zeros(num_layers, batch_size, hidden_dim).to(device),
                torch.zeros(num_layers, batch_size, hidden_dim).to(device))

# Define Model Parameters
embed_dim = 256
hidden_dim = 512
num_layers = 2
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
def load_model():
    model = ShayariLSTM(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
    model.load_state_dict(torch.load("shayari_lstm_final.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Function to format generated text into shayari format
def format_shayari(text):
    # Add a newline after punctuation marks for better readability
    text = re.sub(r'([.!?,])', r'\1\n', text)
    # Break lines every 5-7 words for a poetic structure
    words = text.split()
    formatted_lines = [' '.join(words[i:i+6]) for i in range(0, len(words), 6)]
    return '\n'.join(formatted_lines)

# Generate text function
def generate_text(start_seq, gen_length=300):
    tokens = [token_to_id.get(char, 0) for char in start_seq]
    input_seq = torch.tensor(tokens).unsqueeze(0).to(device)
    hidden = model.init_hidden(1, hidden_dim, num_layers, device)
    generated = tokens

    with torch.no_grad():
        for _ in range(gen_length):
            output, hidden = model(input_seq, hidden)
            prob = torch.softmax(output[:, -1, :], dim=-1)
            next_token = torch.argmax(prob, dim=-1).item()

            generated.append(next_token)
            input_seq = torch.tensor([[next_token]]).to(device)

    id_to_token = {v: k for k, v in token_to_id.items()}
    generated_text = ''.join([id_to_token.get(token, '') for token in generated])
    
    return format_shayari(generated_text)

# Streamlit UI
st.title("🌟 UrduGenShayar 🌟")
st.write("Tum alfaaz likhu tumhari shaayari banayein")

# User input
start_seq = st.text_input("📝 Bayaan Karo:", " ")

if st.button("✨ Generate Shayari ✨"):
    generated_shayari = generate_text(start_seq, gen_length=300)
    st.subheader("🎤 Generated Shayari:")
    st.text_area("", generated_shayari, height=250)
