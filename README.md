# 📝 Roman Urdu Poetry Generator using LSTM  

## 🌟 Introduction  
In the world of AI-generated content, large language models like GPT have revolutionized text generation. But what if we could achieve impressive results without relying on transformer-based architectures? This project explores how **Long Short-Term Memory (LSTM)** networks can be leveraged to generate **Roman Urdu poetry**, using a dataset scraped from [Rekhta.org](https://rekhta.org).  

## 📌 Project Overview  
This repository contains an LSTM-based generative model trained on Roman Urdu poetry. By tokenizing the text using **Byte Pair Encoding (BPE) and Character Pair Encoding**, the model learns to generate meaningful poetic lines while preserving the linguistic beauty of Urdu.  

## 🚀 Features  
- 📜 **Roman Urdu Poetry Generation**  
- 🧠 **Deep Learning with LSTM**  
- 🔤 **Custom Tokenization: BPE & Character-Pair Encoding**  
- 📈 **Trained with PyTorch for Efficient Learning**  
- 🎭 **Captures Contextual Patterns in Poetry**  

## 🔧 Installation  
Clone the repository and install the dependencies:  

```bash
git clone https://github.com/mkhizu21/UrduGenShayar.git  
cd roman-urdu-poetry-generator  
pip install -r requirements.txt

> **Preview Directly In Streamlit
> https://urdugenshayar-9171-9301.streamlit.app/
> **

> **To Run Locally on your PC, Clone the Repository and Run the following command in Terminal
> python -m streamlit run app.py 
> **
 
📂 Dataset

The dataset consists of Roman Urdu poetry scraped from Rekhta. Preprocessing includes:
![WhatsApp Image 2025-02-16 at 14 44 43_cde68f4c](https://github.com/user-attachments/assets/2e1e13b6-8133-41cd-8aed-5ae0e749e57c)
✔ Cleaning & Normalization
✔ Tokenization (BPE & Character-Pair Encoding)
✔ Sequence Preparation for LSTM Training

🏗️ Model Architecture
The model is built using a stacked LSTM architecture with PyTorch:

python train.py  
Adjust hyperparameters in config.py:

EPOCHS = 100
LSTM_LAYERS = 4
TOKENIZER = 'character-pair'
🎤 Poetry Generation
Generate poetry using:

python generate.py --seed "tujh pe uthi hain" --length 50

Example Output:

[WhatsApp Image 2025-02-16 at 14 44 43_cde68f4c](https://github.com/user-attachments/assets/aeb23ee5-714a-4d3b-9997-ea67ce5fd6cd)


📊 Results & Observations

BPE vs Character-Pair Encoding: BPE performed well but struggled with small datasets. Character-pair encoding produced better results despite lower accuracy.
Epochs & LSTM Layers: Increasing epochs from 50 to 100 and LSTM layers from 2 to 4 improved coherence.
Dataset Limitation: The model generated fluent text, but a larger dataset would enhance quality further.

##🔮 Future Enhancements

📈 Train on a Larger Corpus for Better Contextual Understanding
🔀 Experiment with GRU & Transformer Architectures
🌐 Deploy as a Web API for Interactive Usage
🤝 Contributing
Contributions are welcome! If you have ideas to improve the model, feel free to submit a PR.

##📜 License
This project is licensed under the MIT License.

##💌 Connect
Let's talk AI & poetry! Reach out via:
📧 Email: your.email@example.com
🐦 Twitter: @yourhandle
