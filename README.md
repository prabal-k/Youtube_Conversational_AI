# Youtube AI Assistant

lets users upload YouTube URLs and instantly chat with the video's content. Under the hood, it uses a RAG (Retrieval-Augmented Generation) pipeline enhanced with LangGraph memory, ensuring smooth, context-aware conversations across past few interactions. Whether summarizing videos or answering questions, the system keeps track of the discussion for a seamless experience.

---
## Prerequisites

- Python 3.9 or higher
- [Create Groq API Key]( https://console.groq.com/keys)

---
## System Architecture

![Image](https://github.com/user-attachments/assets/0e394ccf-95f5-4c8e-9b64-71d3641573b4)

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/prabal-k/Youtube_Conversational_AI
```

### 2. Open with VsCode ,Create and Activate a Python Virtual Environment

### On Windows:
```
python -m venv venv

venv\Scripts\activate
```
### On Linux/macOS:
```
python3 -m venv venv

source venv/bin/activate
```
### 3. Install Required Dependencies
``
pip install -r requirements.txt
``
### 4. Configure Environment Variables

Create a .env file in the root folder with the following content:

GROQ_API_KEY = "your_groq_api_key_here"

### 5. Run the Application
``
streamlit run app.py
``

---

## Snapshots

### This demonstrates the Application along with the friendly user interface.

![Image](https://github.com/user-attachments/assets/ab7bb3f6-6e4c-4693-b606-e125257ee981)

---


![Image](https://github.com/user-attachments/assets/2cf0bb65-9ab5-4692-bdde-4f854c2ee293)

---


![Image](https://github.com/user-attachments/assets/54fbddf1-c9bf-489b-addb-1364cab1ca0e)

![Image](https://github.com/user-attachments/assets/9feaf6da-b787-4158-b802-66fc4731fed2)

---

![Image](https://github.com/user-attachments/assets/596cddc1-4415-4167-84a6-5f77960e7e1d)

