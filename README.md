# 📄 Resume Parser API
The Resume Parser API leverages Google AI services to analyze and extract data from resumes. This document guides you through setting up and testing the project on your local machine.

## 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 📋 Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.6 or higher 🐍
- Uvicorn 🦄
- FastAPI 🌟
- Google GenAI 🧠
- PDF Miner 🔍

### 🛠 Installation 
1. **Clone the Repository**
   Start by cloning the repository to your local machine:
   ```bash
   git clone [URL to Repository]
   cd [Repository Name]
   ````
2. **Google AI API Key**
  Obtain a Google AI API key to use Google's AI services:
  - Visit [Google AI Studio](https://aistudio.google.com/ "Visit Google AI Studio") and follow the process to get your API key.

3. **Configure API Key**
  Insert your API key into the cred.env file to enable the API services:
  GOOGLE_AI_API_KEY='YOUR_API_KEY_HERE'

### 🏃 Running the Application
To launch the API server, execute the following command within the directory where you've set up the project:
   ```bash
   python -m uvicorn resume_parser_api:app --reload
   ````

### 📚 Accessing the Documentation
If deployed locally on your machine, you can access the API documentation by visiting
   ```bash
   http://127.0.0.1:8000/docs
   ````

### 🧪 Testing the API
To start making client requests, run the client script included with the project:
Ensure to have a sample resume on your working directory and name it "sample_resume.pdf"
   ```bash
   python resume_parser_client_sample.py
   ````
