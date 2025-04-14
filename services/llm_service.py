# backend/services/llm_service.py
import os
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, LANGCHAIN_MODEL, CATEGORY
import json

# temp set to 0 for deterministic output
llm_keyword_category = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0
)

def extract_keywords_and_category(chat_context: str) -> dict:
    """
    Extract keywords and determine the category (DEV_DOC or MEETING_DOC) from the input chat context.
    This function utilizes an LLM chain with a custom prompt.
    
    Args:
        chat_context (str): The input text to analyze.
    
    Returns:
        dict: A dictionary with 'keywords' (list) and 'category' (str).
    
    Raises:
        Exception: If the LLM call or JSON parsing fails.
    """
    # Define a prompt template to guide the LLM
    prompt_template = PromptTemplate(
        input_variables=["chat_context"],
        template=(
            "아래 내용을 기반으로 핵심 키워드를 추출하고, 기술문서이면 '{dev}', 회의록이면 '{meeting}'로 카테고리를 분류하세요.\n"
            "내용: {chat_context}\n"
            "출력 형식 (JSON): {{\"keywords\": [키워드 목록], \"category\": \"{dev}\" | \"{meeting}\"}}\n"
            "예시: {{\"keywords\": [\"API\", \"JWT\"], \"category\": \"{dev}\"}}."
        )
    )

    formatted_prompt = prompt_template.format(
        chat_context=chat_context,
        dev=CATEGORY.DEV_DOC,  # Dynamically pass dev category
        meeting=CATEGORY.MEETING_DOC  # Dynamically pass meeting category
    )
    
    try:
        # Call the LLM
        response = llm_keyword_category.invoke(formatted_prompt).content
        
        # Parse the JSON response
        result = json.loads(response)
        
        return result
    except Exception as e:
        logging.exception("Error in extract_keywords_and_category")
        raise Exception("LLM 키워드/카테고리 추출 실패") from e


llm_summary = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0.3
)

def generate_document_summary(chat_context: str, category: str) -> dict:
    """
    Generate a summary and create a full document from the chat context.
    The document style is adjusted based on the category.
    
    Args:
        chat_context (str): The input text to summarize.
        category (str): The document category (DEV_DOC or MEETING_DOC).
    
    Returns:
        dict: A dictionary with 'title', 'summary', and 'document'.
    
    Raises:
        Exception: If the LLM call or JSON parsing fails.
    """
    # Choose document style based on category
    if category == CATEGORY.DEV_DOC:
        doc_style = "기술문서"
    else:
        doc_style = "회의록"
    
    # 최종 합의된 내용을 출력하도록
    prompt = PromptTemplate(
        input_variables=["chat_context", "doc_style"],
        template=(
            "아래 채팅 내용을 기반으로 {doc_style}를 작성하세요.\n"
            "채팅 내용: {chat_context}\n"
            "적절한 제목, 문서 요약, 그리고 생성된 전체 문서를 JSON 형식으로 출력하세요.\n"
            "출력 예시:\n"
            "{{\n"
            "  \"title\": \"문서 제목\",\n"
            "  \"summary\": \"문서 요약\",\n"
            "  \"document\": \"전체 생성된 문서 원문\"\n"
            "}}"
        )
    )
    
    formatted_prompt = prompt.format(chat_context=chat_context, doc_style=doc_style)
    
    try:
        # Call the LLM
        response = llm_summary.invoke(formatted_prompt).content
        result = json.loads(response)
        return result
    except Exception as e:
        logging.exception("Error in generate_document_summary")
        raise Exception("LLM 문서 생성 실패") from e