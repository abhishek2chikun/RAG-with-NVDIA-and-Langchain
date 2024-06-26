RAG: Retrieval-Augmented Generation with NVIDIA and Langchain
-------------------------------------------------------------

This application empowers users to upload PDF documents and leverage the power of Retrieval-Augmented Generation (RAG) to ask and receive insightful answers directly from those documents. The core functionalities are underpinned by:

*   **NVIDIA AI Endpoints:** Providing access to state-of-the-art large language models (LLMs) and APIs.
    
*   **Langchain:** Facilitating efficient Q&A prompts and document processing workflows.
    
*   **Streamlit:** Enabling a user-friendly and interactive interface.
    

**Key Features:**

*   **Effortless Document Processing:** Upload and store your PDF documents with ease.
    
*   **Local File Management:** Optionally save uploaded documents locally for further analysis.
    
*   **Advanced Vector Embeddings:** Employ NVIDIA's vector embedding capabilities and the FAISS library to optimize search and retrieval.
    
*   **Comprehensive Question Answering:** Pose complex questions related to the uploaded documents, and the RAG model will generate relevant and accurate responses.
    

**Getting Started:**

1.  pip install streamlit langchain\_nvidia\_ai\_endpoints langchain pdfplumber faiss 
    
2.  NVIDIA\_API\_KEY=your\_api\_key\_here
    
    *   Create a .env file in your project directory.
        
    *   Add the following line, replacing your\_api\_key\_here with your actual NVIDIA API key obtained from the NVIDIA Developer Portal:
        
3.  streamlit run app.py

**User Guide:**

1.  **Document Upload:**
    
    *   Click the "Choose PDF files" button to select and upload your desired PDF documents.
        
2.  **Optionally Save Files Locally:**
    
    *   Clicking the "Save Uploaded Files" button will store the uploaded files in a dedicated "Docu" directory for further reference or processing.
        
3.  **Pose Your Questions:**
    
    *   Enter your question pertaining to the uploaded documents in the provided text input box.
        
    *   Click "Enter" to receive an answer generated by the RAG model using the retrieved information.
        

**Code Overview:**

*   save\_uploaded\_files: Handles the local storage of uploaded documents in the "Docu" directory.
    
*   vector\_embedding: Leverages NVIDIA embeddings and FAISS for efficient vector embedding.
    
*   The application integrates the RAG methodology with NVIDIA AI endpoints and Langchain for Q&A prompts, ensuring a robust and informative user experience.
    

**Dependencies:**

*   Streamlit: Streamlines the creation of user interfaces for web applications.
    
*   Langchain: Simplifies Q&A prompts and document processing.
    
*   pdfplumber: Offers efficient text extraction from PDF documents.
    
*   faiss: Provides advanced functionalities for vector embedding and similarity search.
    

**License:**

This project is distributed under the permissive MIT License. Refer to the LICENSE.md file for comprehensive details.
