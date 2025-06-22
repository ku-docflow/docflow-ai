import os
import logging
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LANGCHAIN_MODEL
from prompts.prompts import dev_doc_prompt, meeting_doc_prompt
from utils.error_handler import handle_error


# LLM 인스턴스 생성. temperature를 낮춰서 답변 생성
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0.1
)

def generate_document(chat_context, category, created_at, created_by, organization_id):
    """chat context와 category를 기반으로 문서를 생성. meeting_doc 또는 dev_doc의 프롬프트를 구분함"""
    try:
        if category == "MEETING_DOC":
            formatted_prompt = meeting_doc_prompt.format(
                chat_context=chat_context,
                created_at=created_at,
                created_by=created_by,
                organization_id=organization_id,
                attendees = "Minjun Kim, Jimin Park, Seojun Park, Jisoo Lee" 
            )
        elif category == "DEV_DOC":
            formatted_prompt = dev_doc_prompt.format(
                chat_context=chat_context,
                created_at=created_at,
                created_by=created_by,
                organization_id=organization_id,
                attendees = "Minjun Kim, Jimin Park, Seojun Park, Jisoo Lee" 
            )
        else:
            return {
                "error": "Invalid Category",
                "message": "카테고리 분류에 실패했습니다.",
                "status_code": 400
            }

        response = llm.invoke(formatted_prompt)

        return response.content

    except Exception as e:
        logging.exception("문서 생성 중 오류 발생")
        return handle_error(
            "Error generating document",
            "문서 생성에 실패했습니다.",
            500
        )
