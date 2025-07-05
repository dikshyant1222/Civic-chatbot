from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file, session
import sys
import os
import re
import time
import json
import uuid
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helper import download_embeddings, extract_data_from_pdf_directory, optimized_text_splitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.chains import ConversationalRetrievalChain
from src.prompt import *
from werkzeug.utils import secure_filename

# Initialize Flask app with correct template folder
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Required for sessions

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Set up environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY

# Memory storage - In production, use Redis or database
conversation_memories = {}

class SessionMemoryManager:
    def __init__(self, memory_type="buffer", max_token_limit=2000, k=10):
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.k = k
        self.last_activity = datetime.now()
        
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif memory_type == "window":
            self.memory = ConversationBufferWindowMemory(
                k=k,
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
        else:
            # Default to buffer memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
    
    def add_message(self, human_message, ai_message):
        """Add a conversation turn to memory"""
        self.memory.save_context(
            {"input": human_message},
            {"answer": ai_message}
        )
        self.last_activity = datetime.now()
    
    def get_memory_variables(self):
        """Get memory variables for the chain"""
        return self.memory.load_memory_variables({})
    
    def get_chat_history(self):
        """Get formatted chat history"""
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        formatted_history = []
        for message in chat_history:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted_history)
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        self.last_activity = datetime.now()
    
    def get_message_count(self):
        """Get the number of messages in memory"""
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        return len(chat_history)

def get_or_create_memory(session_id, memory_type="buffer", max_token_limit=2000, k=10):
    """Get or create conversation memory for a session"""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = SessionMemoryManager(
            memory_type=memory_type,
            max_token_limit=max_token_limit,
            k=k
        )
    return conversation_memories[session_id]

def cleanup_old_memories():
    """Clean up memories older than 24 hours"""
    cutoff_time = datetime.now() - timedelta(hours=24)
    to_remove = []
    
    for session_id, memory_manager in conversation_memories.items():
        if memory_manager.last_activity < cutoff_time:
            to_remove.append(session_id)
    
    for session_id in to_remove:
        del conversation_memories[session_id]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot"
embeddings = download_embeddings()

# Initialize vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name="chatbot",
    embedding=embeddings,
    namespace="default"
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 12})

from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Get Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM with Groq
llm = ChatOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    model="qwen/qwen3-32b",
    temperature=0.3,
    max_completion_tokens=3050,
    top_p=0.95,
    reasoning_effort="default",
    stop=None,
)

# Updated prompt template that includes conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""{system_prompt}

Previous conversation history:
{{chat_history}}

Use the conversation history to provide contextual responses and maintain continuity in the conversation.
If the user refers to something mentioned earlier, acknowledge it appropriately."""),
    ("human", "{input}"),
])

# Create a custom chain that works with memory
def create_conversational_rag_chain(retriever, llm, prompt):
    """Create a conversational RAG chain that works with memory"""
    
    def _combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_chat_history(chat_history):
        if not chat_history:
            return ""
        
        formatted = []
        for message in chat_history:
            if isinstance(message, HumanMessage):
                formatted.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted)
    
    def run_chain(input_data):
        # Get relevant documents
        docs = retriever.get_relevant_documents(input_data["input"])
        
        # Format chat history
        chat_history = _format_chat_history(input_data.get("chat_history", []))
        
        # Prepare the prompt
        formatted_prompt = prompt.format(
            input=input_data["input"],
            chat_history=chat_history,
            context=_combine_docs(docs)
        )
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        
        return {
            "answer": response.content,
            "source_documents": docs
        }
    
    return run_chain

# Create the conversational RAG chain
conversational_rag_chain = create_conversational_rag_chain(retriever, llm, prompt)

@app.route("/")
def home():
    # Generate a session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Clean up old memories periodically
    cleanup_old_memories()
    
    return render_template("chat.html")

# Helper to strip out <think> tags
def clean_response(text):
    # Remove <think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Ensure section headers start on a new line and are followed by a line break
    text = re.sub(r"\s*(Answer:|Legal Basis:|Context:|Follow-up:)\s*", r"\n\1\n", text)
    # Add a blank line between sections for clarity
    text = re.sub(r"\n(Answer:|Legal Basis:|Context:|Follow-up:)\n", r"\n\n\1\n", text)
    # Remove excessive blank lines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace
    return text.strip()

@app.route("/get_response", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response_type = request.form.get("response_type", "normal")
    memory_type = request.form.get("memory_type", "buffer")  # "buffer" or "window"
    
    # Get session ID
    session_id = session.get('session_id', str(uuid.uuid4()))
    if 'session_id' not in session:
        session['session_id'] = session_id
    
    # Get conversation memory
    memory_manager = get_or_create_memory(session_id, memory_type=memory_type)
    
    # Build input message with any response type instructions
    input_message = msg
    if response_type == "short":
        input_message = f"{msg} (Please provide a very brief response)"
    elif response_type == "detailed":
        input_message = f"{msg} (Please provide a detailed explanation)"
    
    try:
        # Get memory variables
        memory_vars = memory_manager.get_memory_variables()
        
        # Prepare input for the chain
        chain_input = {
            "input": input_message,
            "chat_history": memory_vars.get("chat_history", [])
        }
        
        # Invoke the conversational chain
        response = conversational_rag_chain(chain_input)
        answer = clean_response(response["answer"])
        
        # Add to memory
        memory_manager.add_message(msg, answer)
        
        return answer
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

@app.route("/stream_response", methods=["POST"])
def stream_chat():
    msg = request.form["msg"]
    response_type = request.form.get("response_type", "normal")
    memory_type = request.form.get("memory_type", "buffer")  # "buffer" or "window"
    
    # Get session ID
    session_id = session.get('session_id', str(uuid.uuid4()))
    if 'session_id' not in session:
        session['session_id'] = session_id
    
    # Get conversation memory
    memory_manager = get_or_create_memory(session_id, memory_type=memory_type)
    
    # Build input message with any response type instructions
    input_message = msg
    if response_type == "short":
        input_message = f"{msg} (Please provide a very brief response)"
    elif response_type == "detailed":
        input_message = f"{msg} (Please provide a detailed explanation)"
    
    def generate():
        try:
            # Get memory variables
            memory_vars = memory_manager.get_memory_variables()
            
            # Prepare input for the chain
            chain_input = {
                "input": input_message,
                "chat_history": memory_vars.get("chat_history", [])
            }
            
            # Invoke the conversational chain
            response = conversational_rag_chain(chain_input)
            answer = clean_response(response["answer"])
            
            # Add to memory
            memory_manager.add_message(msg, answer)
            
            for word in answer.split():
                yield word + " "
                time.sleep(0.05)
        except Exception as e:
            print(f"Error in stream_chat: {str(e)}")
            yield "I apologize, but I encountered an error processing your request."
    
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    """Clear conversation memory for the current session"""
    session_id = session.get('session_id')
    if session_id and session_id in conversation_memories:
        conversation_memories[session_id].clear_memory()
        return jsonify({"status": "success", "message": "Memory cleared"})
    return jsonify({"status": "error", "message": "No active session found"})

@app.route("/get_memory", methods=["GET"])
def get_memory():
    """Get conversation history for the current session"""
    session_id = session.get('session_id')
    if session_id and session_id in conversation_memories:
        memory_manager = conversation_memories[session_id]
        chat_history = memory_manager.get_chat_history()
        message_count = memory_manager.get_message_count()
        
        return jsonify({
            "status": "success",
            "chat_history": chat_history,
            "message_count": message_count,
            "memory_type": memory_manager.memory_type
        })
    return jsonify({"status": "error", "message": "No active session found"})

@app.route("/set_memory_type", methods=["POST"])
def set_memory_type():
    """Set the memory type for the current session"""
    session_id = session.get('session_id')
    memory_type = request.form.get("memory_type", "buffer")  # "buffer" or "window"
    k = int(request.form.get("k", 10))  # For window memory
    
    if memory_type not in ["buffer", "window"]:
        return jsonify({"status": "error", "message": "Invalid memory type. Use 'buffer' or 'window'"})
    
    try:
        if session_id:
            # Create new memory manager with specified type
            conversation_memories[session_id] = SessionMemoryManager(
                memory_type=memory_type,
                k=k
            )
            
            return jsonify({
                "status": "success", 
                "message": f"Memory type set to {memory_type}",
                "k": k if memory_type == "window" else None
            })
        
        return jsonify({"status": "error", "message": "No active session found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/get_memory_info", methods=["GET"])
def get_memory_info():
    """Get detailed memory information"""
    session_id = session.get('session_id')
    if session_id and session_id in conversation_memories:
        memory_manager = conversation_memories[session_id]
        
        return jsonify({
            "status": "success",
            "memory_type": memory_manager.memory_type,
            "message_count": memory_manager.get_message_count(),
            "k": memory_manager.k if memory_manager.memory_type == "window" else None,
            "max_token_limit": memory_manager.max_token_limit
        })
    return jsonify({"status": "error", "message": "No active session found"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)