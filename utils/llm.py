# -*- coding: utf-8 -*-
"""
utils/llm.py
Integration with Mistral API for Retrieval-Augmented Generation.
"""

import os
import sys
from typing import Optional

# Set UTF-8 encoding for stdout/stderr
if sys.version_info[0] < 3:
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr, 'strict')


def generate_answer(query: str, context: str, model: str = "mistral-small-latest") -> str:
    """
    Generate an answer using Mistral API with RAG context.
    
    Args:
        query: User's question
        context: Retrieved document context
        model: Mistral model to use
        
    Returns:
        Generated answer string
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        return "Please set your Mistral API key in the sidebar to generate answers."
    
    try:
        # Try different package names
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai_sdk import Mistral
        
        client = Mistral(api_key=api_key)
        
        # Build the prompt with context
        prompt = f"""Based on the following context, please answer the user's question. 
If the answer cannot be determined from the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Call Mistral API
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return "Error: mistralai package not installed. Run: pip install mistralai"
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def generate_interview_questions(resume_context: str, interview_type: str = "Technical") -> str:
    """
    Generate interview questions based on resume content.
    
    Args:
        resume_context: Extracted resume text chunks
        interview_type: Type of interview (Technical, Behavioral, Mixed, HR Round)
        
    Returns:
        Generated interview questions
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        return "Please set your Mistral API key in the sidebar to generate questions."
    
    try:
        from mistralai import Mistral
    except ImportError:
        try:
            from mistralai_sdk import Mistral
        except ImportError:
            return "Error: mistralai package not installed. Run: pip install mistralai"
    
    client = Mistral(api_key=api_key)
    
    prompt = f"""Based on the following resume content, generate 8 {interview_type} interview questions.
The questions should be relevant to the candidate's skills, experience, and projects mentioned in the resume.

Resume Content:
{resume_context}

Generate 8 {interview_type} interview questions. Number them 1-8."""
        
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return "Error: mistralai package not installed. Run: pip install mistralai"
    except Exception as e:
        return f"Error generating questions: {str(e)}"


def summarize_context(context: str, topic: Optional[str] = None) -> str:
    """
    Summarize the provided context.
    
    Args:
        context: Document context to summarize
        topic: Optional specific topic to focus on
        
    Returns:
        Summary text
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        return "Please set your Mistral API key in the sidebar to summarize."
    
    try:
        from mistralai import Mistral
        
        client = Mistral(api_key=api_key)
        
        topic_prefix = f" about {topic}" if topic else ""
        
        prompt = f"""Please provide a concise summary of the following content{topic_prefix}.

Content:
{context}

Summary:"""
        
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return "Error: mistralai package not installed. Run: pip install mistralai"
    except Exception as e:
        return f"Error generating summary: {str(e)}"
