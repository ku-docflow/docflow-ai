# backend/services/llm_service.py
import os
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, LANGCHAIN_MODEL, CATEGORY
import json

# temp set to 0 for deterministic output
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0
)

def extract_keywords_and_category(chat_context: str) -> dict:
    """
    Extract keywords and determine the category (DEV_DOC or MEETING_DOC) from the input chat context.
    """
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
        dev=CATEGORY.DEV_DOC,
        meeting=CATEGORY.MEETING_DOC 
    )
    
    try:
        # Call the LLM
        response = llm.invoke(formatted_prompt).content
        
        # Parse the JSON response
        result = json.loads(response)
        
        return result
    except Exception as e:
        logging.exception("Error in extract_keywords_and_category")
        raise Exception("LLM 키워드/카테고리 추출 실패") from e
