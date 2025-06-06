import os
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, LANGCHAIN_MODEL, CATEGORY
import json
from utils.error_handler import handle_error

# LLM 인스턴스 생성. temperature를 낮춰서 답변 생성
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0.1
)

def generate_document_summary(chat_context: str, category: str) -> dict:
    """
    Chat context 를 기반으로 문서 요약을 생성
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
        response = llm.invoke(formatted_prompt).content
        result = json.loads(response)
        return result
    except Exception as e:
        logging.exception("Error in generate_document_summary")
        return handle_error(
            "Error generating document summary",
            "문서 요약 생성에 실패했습니다.",
            500
        )