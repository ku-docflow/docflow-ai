import os
import logging
from langchain_community.chat_models import ChatOpenAI
from config import OPENAI_API_KEY, LANGCHAIN_MODEL
from prompts.prompts import dev_doc_prompt, meeting_doc_prompt

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0.1
)

def generate_document(chat_context, category, created_at, created_by, organization_id):
    try:
        if category == "MEETING_DOC":
            formatted_prompt = meeting_doc_prompt.format(
                chat_context=chat_context,
                created_at=created_at,
                created_by=created_by,
                organization_id=organization_id
            )
        elif category == "DEV_DOC":
            formatted_prompt = dev_doc_prompt.format(
                chat_context=chat_context,
                created_at=created_at,
                created_by=created_by,
                organization_id=organization_id
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
        return {
            "error": "Document Generation Error",
            "message": str(e),
            "status_code": 500
        }
