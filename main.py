
import os
import boto3
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import openai
import logging
from typing import List, Optional
import time
import uuid

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced NOAH API", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced models
class Memory(BaseModel):
    user_id: str
    text: str
    tags: Optional[List[str]] = Field(default_factory=list)
    sentiment: Optional[str] = None
    category: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str
    context_window: Optional[int] = 5

# Environment validation
required_env_vars = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_S3_BUCKET_NAME": os.getenv("AWS_S3_BUCKET_NAME")
}

for var_name, var_value in required_env_vars.items():
    if not var_value:
        raise ValueError(f"{var_name} environment variable is not set")

# Initialize clients
openai.api_key = required_env_vars["OPENAI_API_KEY"]
s3_client = boto3.client(
    's3',
    aws_access_key_id=required_env_vars["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=required_env_vars["AWS_SECRET_ACCESS_KEY"]
)
bucket_name = required_env_vars["AWS_S3_BUCKET_NAME"]

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
def root():
    return {
        "message": "NOAH is alive!",
        "version": "2.0",
        "status": "operational"
    }

async def analyze_sentiment(text: str) -> str:
    """Analyze text sentiment using OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Analyze the sentiment of the following text and respond with only one word: POSITIVE, NEGATIVE, or NEUTRAL"
            }, {
                "role": "user",
                "content": text
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return "NEUTRAL"

async def categorize_text(text: str) -> str:
    """Categorize text using OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Categorize the following text into one of these categories: PERSONAL, WORK, EDUCATION, HEALTH, OTHER"
            }, {
                "role": "user",
                "content": text
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Categorization failed: {str(e)}")
        return "OTHER"

@app.post("/store_memory/")
async def store_memory(memory: Memory):
    try:
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        file_key = f"memories/{memory.user_id}.json"
        
        # Enhance memory with AI analysis
        sentiment = await analyze_sentiment(memory.text)
        category = await categorize_text(memory.text)
        
        memory_data = {
            "id": memory_id,
            "text": memory.text,
            "timestamp": timestamp,
            "tags": memory.tags,
            "sentiment": sentiment,
            "category": category
        }
        
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            existing_data = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            existing_data = []
        
        existing_data.append(memory_data)
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=json.dumps(existing_data),
            ContentType='application/json'
        )

        return {
            "message": "Memory saved",
            "memory_id": memory_id,
            "timestamp": timestamp,
            "analysis": {
                "sentiment": sentiment,
                "category": category
            }
        }
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store memory")

@app.get("/retrieve_memory/{user_id}")
async def retrieve_memory(
    user_id: str,
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: Optional[int] = None
):
    try:
        file_key = f"memories/{user_id}.json"
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            memories = json.loads(response['Body'].read().decode('utf-8'))
            
            # Apply filters
            if category:
                memories = [m for m in memories if m.get("category") == category]
            if sentiment:
                memories = [m for m in memories if m.get("sentiment") == sentiment]
            
            # Apply limit
            if limit:
                memories = memories[-limit:]
                
            return {"memories": memories}
        except s3_client.exceptions.NoSuchKey:
            return {"memories": []}
    except Exception as e:
        logger.error(f"Error retrieving memories: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memories")

@app.post("/chat/")
async def chat_with_noah(chat_request: ChatRequest):
    try:
        # Retrieve recent memories for context
        memories = await retrieve_memory(
            chat_request.user_id,
            limit=chat_request.context_window
        )
        
        # Build context from memories
        context = "\n".join([
            f"Previous memory: {m['text']}"
            for m in memories.get("memories", [])
        ])
        
        messages = [
            {
                "role": "system",
                "content": f"You are NOAH, a helpful AI assistant. Here is some context about our previous interactions:\n{context}"
            },
            {
                "role": "user",
                "content": chat_request.message
            }
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        
        return {
            "response": response.choices[0].message.content,
            "context_used": bool(context)
        }
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate chat response")

@app.get("/analytics/{user_id}")
async def get_analytics(user_id: str):
    """Get analytics about user's memories."""
    try:
        memories = (await retrieve_memory(user_id)).get("memories", [])
        
        # Calculate analytics
        total_memories = len(memories)
        sentiment_distribution = {}
        category_distribution = {}
        
        for memory in memories:
            sentiment = memory.get("sentiment", "UNKNOWN")
            category = memory.get("category", "UNKNOWN")
            
            sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        return {
            "total_memories": total_memories,
            "sentiment_distribution": sentiment_distribution,
            "category_distribution": category_distribution
        }
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics")

@app.delete("/memory/{user_id}/{memory_id}")
async def delete_memory(user_id: str, memory_id: str):
    """Delete a specific memory."""
    try:
        file_key = f"memories/{user_id}.json"
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            memories = json.loads(response['Body'].read().decode('utf-8'))
            
            # Filter out the memory to delete
            updated_memories = [m for m in memories if m.get("id") != memory_id]
            
            if len(updated_memories) == len(memories):
                raise HTTPException(status_code=404, detail="Memory not found")
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=file_key,
                Body=json.dumps(updated_memories)
            )
            
            return {"message": "Memory deleted successfully"}
        except s3_client.exceptions.NoSuchKey:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete memory")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint."""
    try:
        # Test S3 connection
        s3_client.list_buckets()
        # Test OpenAI connection
        openai.Model.list()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "s3": "operational",
                "openai": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
