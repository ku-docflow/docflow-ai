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
            "다음 내용을 기반으로 핵심 키워드를 추출하고, 기술문서면 '{dev}', 회의록이면 '{meeting}'로 분류하세요.\n"
            "- 반드시 JSON 형식만 출력하세요.\n"
            "- 설명 없이 출력만 하세요.\n"
            "- 예: {{\"keywords\": [\"API\", \"JWT\"], \"category\": \"{dev}\"}}\n\n"
            "내용:\n{chat_context}"
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
