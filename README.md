# YouTube-Video-Chat

This repository contains a Streamlit application that enables users to chat with YouTube videos. The application processes video transcripts to provide accurate and contextual answers to user queries. It also maintains chat history to enhance the conversational experience.

## Features
- **Video Transcript Processing**: Extracts and processes the transcript from YouTube videos.
- **Contextual Q&A**: Provides detailed answers based on the video content.
- **Chat History**: Maintains chat history for better context in conversations.
- **Streamlit Interface**: User-friendly interface for easy interaction.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/YourUsername/YouTube-Video-Chat.git
    ```
2. Navigate to the project directory:
    ```sh
    cd YouTube-Video-Chat
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Set up your environment variables:
    - Create a `.env` file in the project directory.
    - Add your Google API key:
      ```plaintext
      GOOGLE_API_KEY=your_google_api_key
      ```

## Usage
1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```
2. In the sidebar, enter the URL of the YouTube video and click "Process".
3. Enter your question in the provided input field and click "Submit".
4. The application will provide answers based on the video content and display the chat history.

## Code Overview
### Main Components
- **get_text(url)**: Extracts the transcript from a YouTube video.
- **get_chunks(text)**: Splits the transcript into manageable chunks.
- **get_retriever(chunks)**: Creates a retriever using FAISS and Google Generative AI embeddings.
- **get_prompt()**: Generates a prompt template for the Q&A chain.
- **process_url(url)**: Processes the YouTube URL to obtain the retriever.
- **get_history(retriever)**: Creates a history-aware retriever to maintain chat history.
- **Streamlit Interface**: Provides a user-friendly interface to interact with the application.


## Acknowledgments
- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Generative AI](https://cloud.google.com/ai/generative-ai)
- [Streamlit](https://streamlit.io/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

