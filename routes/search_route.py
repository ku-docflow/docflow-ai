from flask import Blueprint, request, jsonify
import logging
from services import document_service, summary_service, qdrant_service
from utils.error_handler import handle_error
from prompts.prompts import summary_prompt, answer_prompt

search_bp = Blueprint("search", __name__)
logger = logging.getLogger(__name__)

@search_bp.route("/search-document", methods=["POST"])
def search_document():
    """
    POST endpoint to perform RAG-style 문서 검색 및 응답 생성.
    Expected JSON payload:
    {
        "references": [
            { "title": "제목", "content": "Markdown 내용" },
            { "title": "제목", "content": "Markdown 내용" },
            { "title": "제목", "content": "Markdown 내용" }
        ],
        "userQuery": "질문 텍스트"
    }
    """
    try:
        # 1) JSON 추출 및 필수 필드 검증
        data = request.get_json(force=True)
        references = data.get("references")
        user_query = data.get("userQuery")

        if not references or not user_query:
            return handle_error(
                "Missing Fields",
                "references 또는 userQuery가 누락되었습니다.",
                400
            )

        if len(references) != 3:
            return handle_error(
                "Invalid References",
                "3개의 참조 문서를 제공해야 합니다.",
                400
            )

        # 2) 문서별 요약 생성
        summarized_docs = []
        for ref in references:
            title = ref.get("title")
            content = ref.get("content")

            if not title or not content:
                return handle_error(
                    "Invalid Reference Item",
                    "각 참조에는 title과 content가 모두 포함되어야 합니다.",
                    400
                )

            summary = summary_service.summarize_content(
                content.strip(),
                summary_prompt
            )
            summarized_docs.append(f"# {title}\n{summary}")

        # 3) 요약된 문서 합치고 RAG 응답 생성
        combined_summary = "\n\n".join(summarized_docs)
        rag_response = document_service.answer_question_with_summary(
            combined_summary,
            user_query.strip(),
            answer_prompt
        )

        # 4) 결과 반환
        return jsonify({
            "statusCode": 200,
            "message": "성공했습니다",
            "data": {
                "ragResponse": rag_response
            }
        }), 200

    except Exception as e:
        logger.exception("Error in /search-document")
        return handle_error(
            "/search-document failed",
            "RAG 응답 생성에 실패했습니다.",
            500
        )
