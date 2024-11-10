Document Processing and Q&A System

Table of Contents

	1.	Overview
	2.	Features
	3.	Prerequisites
	4.	Installation
	•	Using Conda
	•	Using Pip
	•	Combined Installation
	5.	Configuration
	6.	Directory Structure
	7.	Usage
	•	Starting the Application
	•	API Endpoints
	•	Home Page
	•	Contact Page
	•	List Processed Files
	•	Process File
	•	WebSocket Endpoints
	•	Llama Q&A WebSocket
	•	OpenAI Realtime API WebSocket
	8.	Functional Components
	•	Embedding Generation
	•	Vector Store Index Initialization
	•	Document Processing
	•	PDF Processing
	•	PPTX Processing
	•	Image Processing
	•	Audio Processing
	•	Text Extraction
	•	Retrieval-Augmented Generation (RAG)
	•	Completion Generation
	9.	Logging
	10.	Error Handling
	11.	Environment Variables
	12.	Audio File Limitations
	13.	Dependencies
	14.	License
	15.	Contact

Overview

The Document Processing and Q&A System is a comprehensive application built with FastAPI that facilitates the uploading, processing, and querying of various document types. It leverages advanced technologies and APIs, including NVIDIA’s Embedding and Llama models, Groq’s vision and transcription models, and OpenAI’s Realtime API. The system supports real-time question-answering via WebSocket connections and offers robust text extraction and indexing capabilities.

Features

	•	File Processing: Supports PDF, PPTX, TXT, RTF, DOC, DOCX, ODT, images (jpg, jpeg, png, gif, bmp, tiff, webp), and audio files (mp3, mp4, mpeg, mpga, m4a, wav, webm).
	•	Text Extraction: Utilizes OCR (pytesseract) and Groq’s advanced vision models for extracting text from images and PDFs.
	•	Embedding Generation: Generates text embeddings using NVIDIA’s Embedding API.
	•	Vector Store Indexing: Indexes documents for efficient similarity search using llama_index.
	•	Real-time Q&A: Provides real-time question-answering capabilities via WebSocket connections.
	•	Audio Transcription: Transcribes audio files using Groq’s Whisper model.
	•	Web Interface: Offers web pages for home, contact, and listing processed files.
	•	Robust Logging: Comprehensive logging for monitoring and debugging.

Prerequisites

	•	Python: Version 3.8 or higher
	•	Conda: Recommended for managing dependencies and environments
	•	API Keys:
	•	NVIDIA API Keys for Embedding and Llama models
	•	Groq API Key
	•	OpenAI API Key
	•	External Tools:
	•	ffmpeg: Required for audio processing with pydub

Installation

There are multiple ways to install the required dependencies for this application. You can choose between using Conda, Pip, or a combination of both.

Using Conda

	1.	Clone the Repository

git clone https://github.com/AdamQuantum/NVIDIA-Developer-Contest-2024
cd document-processing-app


	2.	Create a Conda Environment

conda create -n doc_qna_env python=3.9
conda activate doc_qna_env


	3.	Install Conda Packages

conda install numpy
conda install httpx
conda install pytesseract
conda install -c pytorch faiss-cpu



Using Pip

	1.	Ensure the Conda Environment is Activated

conda activate doc_qna_env


	2.	Install Pip Packages

pip install -r requirements.txt


	3.	Install Additional Pip Packages

pip install nltk
pip install llama-index-embeddings-nvidia --upgrade --quiet
pip install openai
pip install python-multipart
pip install uvicorn[standard]
pip install websockets
pip install striprtf
pip install groq
pip install pydub
pip install sounddevice



Combined Installation

Alternatively, you can perform all installations sequentially:

# Clone the repository
git clone https://github.com/your-repo/document-processing-app.git
cd document-processing-app

# Create and activate conda environment
conda create -n doc_qna_env python=3.9
conda activate doc_qna_env

# Install conda packages
conda install numpy
conda install httpx
conda install pytesseract
conda install -c pytorch faiss-cpu

# Install pip packages from requirements.txt
pip install -r requirements.txt

# Install additional pip packages
pip install nltk
pip install llama-index-embeddings-nvidia --upgrade --quiet
pip install openai
pip install python-multipart
pip install uvicorn[standard]
pip install websockets
pip install striprtf
pip install groq
pip install pydub
pip install sounddevice

Ensure that ffmpeg is installed on your system. On Ubuntu, you can install it using:

sudo apt-get install ffmpeg

For other operating systems, refer to the FFmpeg installation guide.

Configuration

	1.	Environment Variables
Create a .env file in the root directory and populate it with the necessary API keys:

NVIDIA_API_KEY_EMBEDQA=your_nvidia_embedqa_api_key
NVIDIA_API_KEY_LLAMA=your_nvidia_llama_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key


	2.	Directory Structure
The application uses the following directories:
	•	database/: Stores processed .txt files.
	•	static/: Serves static files.
	•	templates/: Contains HTML templates.
These directories are automatically created if they do not exist.

Directory Structure

document-processing-app/
├── database/
├── static/
├── templates/
├── app.py
├── requirements.txt
├── .env
└── README.md

Usage

Starting the Application

To start the FastAPI application, run:

uvicorn app:app --reload

The application will be accessible at http://127.0.0.1:8000/.

API Endpoints

Home Page

	•	Endpoint: /
	•	Method: GET
	•	Description: Serves the home page.

Example Request

GET / HTTP/1.1
Host: localhost:8000

Example Response

Returns the index.html template.

Contact Page

	•	Endpoint: /contact
	•	Method: GET
	•	Description: Serves the contact page.

Example Request

GET /contact HTTP/1.1
Host: localhost:8000

Example Response

Returns the contact.html template.

List Processed Files

	•	Endpoint: /processed_files
	•	Method: GET
	•	Description: Lists all processed .txt files.

Example Request

GET /processed_files HTTP/1.1
Host: localhost:8000

Example Response

{
  "processed_files": ["document1.txt", "presentation1.txt"],
  "message": "Processed files listed successfully."
}

Process File

	•	Endpoint: /process_file
	•	Method: POST
	•	Description: Uploads and processes a file.

Request Parameters
	•	file: The file to be uploaded. Supported types include PDF, PPTX, TXT, RTF, DOC, DOCX, ODT, images (jpg, jpeg, png, gif, bmp, tiff, webp), and audio files (mp3, mp4, mpeg, mpga, m4a, wav, webm).

Example Request

POST /process_file HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=---BOUNDARY
Content-Length: ...

---BOUNDARY
Content-Disposition: form-data; name="file"; filename="example.pdf"
Content-Type: application/pdf

<file content>
---BOUNDARY--

Example Response

{
  "success": true,
  "message": "File 'example.pdf' processed successfully!"
}

WebSocket Endpoints

Llama Q&A WebSocket

	•	Endpoint: /llama_ws
	•	Description: Enables real-time question-answering using similarity search with the Llama model.

Connection and Communication
	1.	Connect to the WebSocket

const ws = new WebSocket("ws://localhost:8000/llama_ws");


	2.	Send a Question

{
  "type": "question",
  "user_input": "What is the capital of France?"
}


	3.	Receive Answers
The server will stream answers in chunks as they are generated.

OpenAI Realtime API WebSocket

	•	Endpoint: /openai_ws
	•	Description: Facilitates interactions with OpenAI’s Realtime API for audio-based communication.

Connection and Communication
	1.	Connect to the WebSocket

const ws = new WebSocket("ws://localhost:8000/openai_ws");


	2.	Send Audio Chunks

{
  "type": "audio_chunk",
  "audio": "base64_encoded_pcm16_audio_data"
}


	3.	Receive Responses
The server will forward responses from OpenAI to the client, including audio and text responses.

Functional Components

Embedding Generation

Function: get_text_embedding(text: str) -> List[float]

Generates an embedding for the provided text using NVIDIA’s Embedding API. Handles API communication, error checking, and logging.

Vector Store Index Initialization

Function: initialize_vector_store_index()

Initializes the VectorStoreIndex with the custom embedding model and indexes existing documents from the database/ directory.

Document Processing

Handles various file types, extracts text, and indexes content.

PDF Processing

Function: extract_text_from_pdf_with_ocr(pdf_content: bytes, filename: str) -> str

Converts PDF pages to images and extracts text using OCR (pytesseract) and Groq’s vision model.

PPTX Processing

Function: process_pptx(file_content: bytes, filename: str) -> str

Extracts text and tables from PPTX files and processes embedded images for text extraction.

Image Processing

Function: process_image_content(image_content: bytes, source_info: str) -> str

Processes images to extract text using both pytesseract OCR and Groq’s vision model, combining results.

Audio Processing

Function: process_audio_file(file_content: bytes, filename: str) -> str

Transcribes audio files using Groq’s Whisper model, ensuring valid transcription.

Text Extraction

Function: extract_text_with_textract(file_content: bytes, file_extension: str, filename: str) -> str

Extracts text from various document formats using textract and striprtf for RTF files.

Retrieval-Augmented Generation (RAG)

Function: get_relevant_paragraphs(query: str) -> List[str]

Retrieves relevant paragraphs from the indexed documents based on the user’s query using similarity search.

Completion Generation

Function: get_llama_completion(prompt: str) -> str

Generates text completions using NVIDIA’s Llama model via their API, handling request construction and response parsing.

Logging

The application uses Python’s built-in logging module for comprehensive logging. Logs are written to both the console and a file named app.log. Logging levels include DEBUG, INFO, WARNING, ERROR, and CRITICAL.

Configuration:

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

Error Handling

The application incorporates robust error handling mechanisms:
	•	API Key Validation: Checks for the presence of necessary API keys at startup and logs errors if any are missing.
	•	File Processing: Validates file types and sizes, handling unsupported formats gracefully.
	•	WebSocket Communication: Manages disconnections and unexpected errors, ensuring stability.
	•	Function-Level Errors: Catches and logs exceptions within functions to prevent crashes and provide meaningful error messages to users.

Environment Variables

The application relies on the following environment variables, which should be set in a .env file:
	•	NVIDIA_API_KEY_EMBEDQA: API key for NVIDIA’s Embedding API.
	•	NVIDIA_API_KEY_LLAMA: API key for NVIDIA’s Llama model.
	•	GROQ_API_KEY: API key for Groq services.
	•	OPENAI_API_KEY: API key for OpenAI’s Realtime API.

Example .env File:

NVIDIA_API_KEY_EMBEDQA=your_nvidia_embedqa_api_key
NVIDIA_API_KEY_LLAMA=your_nvidia_llama_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key

Audio File Limitations

When processing audio files, the following limitations apply:
	•	Maximum File Size: 25 MB
	•	Minimum File Length: 0.01 seconds
	•	Minimum Billed Length: 10 seconds
	•	If a request is submitted with audio shorter than 10 seconds, it will still be billed for 10 seconds.
	•	Supported File Types: mp3, mp4, mpeg, mpga, m4a, wav, webm
	•	Single Audio Track:
	•	Only the first track will be transcribed for files with multiple audio tracks (e.g., dubbed videos).
	•	Supported Response Formats: json, verbose_json, text

Ensure that audio files adhere to these limitations to avoid processing errors or unexpected billing.

Dependencies

The application utilizes a wide range of Python packages and external tools:
	•	Python Packages:
	•	fastapi: Web framework for building APIs.
	•	uvicorn: ASGI server for running FastAPI applications.
	•	httpx: Asynchronous HTTP client.
	•	faiss-cpu: Library for efficient similarity search and clustering of dense vectors.
	•	pytesseract: OCR tool for text extraction from images.
	•	numpy: Numerical computing library.
	•	pillow: Image processing library.
	•	python-docx: Library for creating and updating Microsoft Word (.docx) files.
	•	python-pptx: Library for creating and updating PowerPoint (.pptx) files.
	•	pdf2image: Converts PDF pages to images.
	•	tiktoken: Tokenizer for splitting text into tokens.
	•	python-dotenv: Loads environment variables from a .env file.
	•	textract==1.6.3: Library for extracting text from various document formats.
	•	jinja2: Templating engine for Python.
	•	nltk: Natural Language Toolkit for text processing.
	•	llama-index-embeddings-nvidia: NVIDIA’s custom embedding models for llama_index.
	•	openai: OpenAI API client.
	•	python-multipart: Parses multipart/form-data which is primarily used for uploading files.
	•	websockets: WebSocket client and server library.
	•	striprtf: Library for stripping RTF formatting.
	•	groq: Groq API client for accessing their services.
	•	pydub: Audio manipulation library.
	•	sounddevice: Audio playback and recording.
	•	External Tools:
	•	ffmpeg: Required for audio processing with pydub.

Installation Example:

pip install fastapi uvicorn httpx faiss-cpu pytesseract numpy pillow python-docx python-pptx pdf2image tiktoken python-dotenv textract==1.6.3 jinja2 nltk llama-index-embeddings-nvidia openai python-multipart websockets striprtf groq pydub sounddevice

Note: Ensure ffmpeg is installed on your system. On Ubuntu, you can install it using:

sudo apt-get install ffmpeg

For other operating systems, refer to the FFmpeg installation guide.

License

This project is licensed under the MIT License.

Contact

For any questions or support, please contact:
	•	Email: adam.martin@email.com
	•	GitHub: https://github.com/AdamQuantum

End of README.md
