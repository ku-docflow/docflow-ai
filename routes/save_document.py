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
    try:
        # Step 1: Parse and validate request data
        data = request.get_json(force=True)
        logger.info(f"Received request data: {data}")

        document_id = int(data.get("documentId"))
        organization_id = int(data.get("organizationId"))
        chat_context = data.get("content")
        user_id = data.get("userId")
        created_by = data.get("createdBy")
        created_at = data.get("createdAt")

        logger.info(f"Parsed data: document_id={document_id}, organization_id={organization_id}, user_id={user_id}")

        # Step 2: Generate and extract keywords and category
        keywords_category = extract_keyword.extract_keywords_and_category(chat_context)
        keywords = keywords_category.get("keywords")
        category = keywords_category.get("category")
        logger.info(f"Extracted keywords: {keywords}, category: {category}")

        # Step 3: Generate summary using LangChain
        summary_doc = generate_summary.generate_document_summary(chat_context, category)
        title = summary_doc.get("title")
        summary = summary_doc.get("summary")
        logger.info(f"Generated summary: title={title}")

        # Step 4: Store document in Qdrant
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

        # Step 5: Return success response
        return jsonify({
            "statusCode": 200,
            "message": "성공했습니다",
        })

    except Exception as e:
        logger.exception("Failed to save document")
        return handle_error("/process-document failed", "LLM 응답 생성에 실패했습니다.", 500)
    