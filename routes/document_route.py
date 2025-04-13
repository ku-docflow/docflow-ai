from flask import Blueprint, request, jsonify
import logging
from services import llm_service, qdrant_service
from utils.error_handler import handle_error

document_bp = Blueprint("document", __name__)

@document_bp.route("/process-document", methods=["POST"])
def process_document():
    """
    POST endpoint to process a document.
    Expected JSON payload:
    {
      "documentId": "uuid",
      "chatContext": "text with chat messages",
      "userId": "user-id",
      "createdBy": "author name"
    }
    """
    try:
        # Extract and validate JSON input
        data = request.get_json(force=True)
        document_id = data.get("documentId")
        chat_context = data.get("chatContext")
        user_id = data.get("userId")
        created_by = data.get("createdBy")
        
        if not all([document_id, chat_context, user_id, created_by]):
            return handle_error("Missing Fields","필수 필드가 누락되었습니다.", 400)
        
        # LLM Call 1: Keyword extraction and category classification
        keywords_category = llm_service.extract_keywords_and_category(chat_context)
        keywords = keywords_category.get("keywords")
        category = keywords_category.get("category")

        print(f"Extracted keywords: {keywords}, Category: {category}")
        
        # LLM Call 2: Generate summary and document content based on category
        summary_doc = llm_service.generate_document_summary(chat_context, category)
        title = summary_doc.get("title")
        document_text = summary_doc.get("document")
        summary = summary_doc.get("summary")
        
        # Qdrant vector store: Chunk the input, generate embeddings, and store the document
        qdrant_service.store_document_embedding(
            document_id,
            {
                "title": title,
                "summary": summary,
                "userId": user_id,
                "createdBy": created_by,
                "keywords": keywords,
                "category": category
            }
        )
        
        # Return response
        return jsonify({
            "statusCode": 200,
            "message": "성공했습니다",
            "data": {
                "documentId": document_id,
                "title": title,
                "document": document_text,
                "userId": user_id,
                "createdBy": created_by,
                "category": category
            }
        })
    except Exception as e:
        logging.exception("Error processing document")
        return handle_error("/process-document failed", "LLM 응답 생성에 실패했습니다.", 500)
