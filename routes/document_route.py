from flask import Blueprint, request, jsonify
import logging
from services import generate_document, generate_summary, extract_keyword, qdrant_service
from utils.error_handler import handle_error
from prompts.prompts import dev_doc_prompt, meeting_doc_prompt

document_bp = Blueprint("document", __name__)

@document_bp.route("/process-document", methods=["POST"])
def process_document():
    """
    @생성봇 기능 : JSON 입력을 받아서 문서 생성, 요약, 키워드 추출 및 Qdrant에 저장하는 엔드포인트

    JSON Payload 예시:
    {
        "documentId": 123,
        "organizationId": 456,
        "userId": 789,
        "chatContext": "여기에 대화 내용이 들어갑니다.",
        "createdBy": "사용자 이름",
        "createdAt": "2023-10-01T12:00:00Z"
    }
    
    응답 예시:
    {
        "statusCode": 200,
        "message": "성공했습니다",
        "data": {
            "documentId": 123,
            "organizationId": 456,
            "title": "문서 제목",
            "document": "문서 내용",
            "summary": "문서 요약",
            "userId": 789,
            "createdBy": "사용자 이름",
            "category": "문서 카테고리",
            "createdAt": "2023-10-01T12:00:00Z"
        }
    }
    """
    try:
        # Extract and validate JSON input
        data = request.get_json(force=True)
        document_id = int(data.get("documentId"))
        organization_id = int(data.get("organizationId"))
        user_id = data.get("userId")
        chat_context = data.get("chatContext")
        created_by = data.get("createdBy")
        created_at = data.get("createdAt", None)
        
        if not all([document_id, chat_context, user_id, created_by, created_at]):
            return handle_error("Missing Fields","필수 필드가 누락되었습니다.", 400)
        
        # LLM Call 1: Keyword extraction and category classification
        keywords_category = extract_keyword.extract_keywords_and_category(chat_context)
        keywords = keywords_category.get("keywords")
        category = keywords_category.get("category")

        print(f"Extracted keywords: {keywords}, Category: {category}")
        

        # LLM Call 2: Generate Document

        full_document = generate_document.generate_document(
            chat_context,
            category,
            created_at,
            created_by,
            organization_id
        )

        # LLM Call 3: Generate summary and document content based on category
        summary_doc = generate_summary.generate_document_summary(chat_context, category)
        title = summary_doc.get("title")
        summary = summary_doc.get("summary")
        
        # Qdrant vector store: Chunk the input, generate embeddings, and store the document
        qdrant_service.store_document_embedding(
            document_id,
            {
                "title": title,
                "summary": summary,
                "document": full_document,
                "userId": user_id,
                "createdBy": created_by,
                "keywords": keywords,
                "category": category,
                "organizationId": organization_id,
                "createdAt": created_at
            }
        )
        
        # Return response to NestJS Server
        return jsonify({
            "statusCode": 200,
            "message": "성공했습니다",
            "data": {
                "documentId": document_id,
                "organizationId": organization_id,
                "title": title,
                "document": full_document,
                "summary": summary,
                "userId": user_id,
                "createdBy": created_by,
                "category": category,
                "createdAt": created_at
            }
        })
    except Exception as e:
        logging.exception("Error processing document")
        return handle_error("/process-document failed", "LLM 응답 생성에 실패했습니다.", 500)
