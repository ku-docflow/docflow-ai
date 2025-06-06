from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import logging
from utils.error_handler import handle_error

logger = logging.getLogger(__name__)
llm = OpenAI(temperature=0)

def answer_question_with_summary(
    summary: str,
    question: str,
    prompt_template: PromptTemplate,
    memory_context: str = ""
) -> str:
    """
    Generate an answer to a question using a summary of documents.
    
    Args:
        summary: The summarized document content
        question: The user's question
        prompt_template: The prompt template to use
        memory_context: Optional context from previous interactions
        
    Returns:
        str: The generated answer
    """
    try:
        # Format the prompt with all context
        formatted_prompt = prompt_template.format(
            summary=summary,
            question=question,
            memory_context=memory_context
        )
        
        # Generate response using the LLM with invoke method
        response = llm.invoke(formatted_prompt)
        return response.strip()
        
    except Exception as e:
        return handle_error("Error generating answer with docs", 500)

def answer_question_without_docs(user_query: str, prompt_template: str) -> str:
    """
    사용자 질문에 대한 답변을 생성합니다.
    """
    try:
        prompt = prompt_template.format(question=user_query)
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return handle_error("Error generating answer without docs", 500)
