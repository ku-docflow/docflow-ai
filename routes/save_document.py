from flask import Blueprint, request, jsonify
import logging
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from services import qdrant_service  # your custom service layer
from services import extract_keyword, generate_summary
from utils.error_handler import handle_error


save_bp = Blueprint("save", __name__)
logger = logging.getLogger(__name__)

@save_bp.route("/save-document", methods=["POST"])
def save_document():
    """
    @문서 저장 엔드포인트 : JSON 입력을 받아서 문서를 저장하고 요약 및 키워드 추출을 수행하는 엔드포인트
    
    JSON Payload 예시:
    {
        "documentId": 123,
        "organizationId": 456,
        "content": "문서 내용",
        "userId": 789,
        "createdBy": "사용자 이름",
        "createdAt": "2023-10-01T12:00:00Z"
    }

    응답 예시:
    {
        "statusCode": 200,
        "message": "성공했습니다"
    }
    """
    try:
        # 1. reqeust 데이터 파싱
        data = request.get_json(force=True)
        logger.info(f"Received request data: {data}")

        document_id = int(data.get("documentId"))
        organization_id = int(data.get("organizationId"))
        chat_context = data.get("content")
        user_id = data.get("userId")
        created_by = data.get("createdBy")
        created_at = data.get("createdAt")

        logger.info(f"Parsed data: document_id={document_id}, organization_id={organization_id}, user_id={user_id}")

        # 2. 키워드 및 카테고리 추출
        keywords_category = extract_keyword.extract_keywords_and_category(chat_context)
        keywords = keywords_category.get("keywords")
        category = keywords_category.get("category")
        logger.info(f"Extracted keywords: {keywords}, category: {category}")

        # 3. summary 생성
        summary_doc = generate_summary.generate_document_summary(chat_context, category)
        title = summary_doc.get("title")
        summary = summary_doc.get("summary")
        logger.info(f"Generated summary: title={title}")

        # 4. qdrant에 문서 임베딩 저장
        qdrant_service.store_document_embedding(
            document_id,
            {
                "title": title,
                "summary": summary,
                "document": chat_context,
                "userId": user_id,
                "createdBy": created_by,
                "keywords": keywords,
                "category": category,
                "organizationId": organization_id,
                "createdAt": created_at
            }
        )
        logger.info("Successfully stored document in Qdrant")

        # 5. 성공했을 경우 응답
        return jsonify({
            "statusCode": 200,
            "message": "성공했습니다",
        })

    except Exception as e:
        logger.exception("Failed to save document")
        return handle_error("/process-document failed", "LLM 응답 생성에 실패했습니다.", 500)
    