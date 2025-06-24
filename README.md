ECHO is on.
# 🌾 AI Crop Disease Classifier

This project helps Nigerian farmers easily detect crop diseases using AI. It uses a pre-trained model from Hugging Face to classify plant diseases based on uploaded images.



## 🔍 Features

- 🖼 Upload an image of a crop leaf
- 🤖 Uses Hugging Face Vision Transformer to detect diseases
- 🔊 Speaks out the result using the browser's speech engine
- 🌐 Fully browser-based frontend
- 🛠 Node.js backend handles file upload and AI communication



## 📸 Demo

🔗 [Live Demo on Render](https://your-render-url.onrender.com)



## 🧰 Tech Stack

- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Backend**: Node.js + Express
- **AI Model**: [Hugging Face - nateraw/vit-base-patch16-224-inaturalist](https://huggingface.co/nateraw/vit-base-patch16-224-inaturalist)
- **Deployment**: Render
- **Speech Output**: Web Speech API (SpeechSynthesis)



## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/crop-classifier.git
cd crop-classifier
