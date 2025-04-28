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
    사용자의 쿼리와 참조 문서들을 기반으로 질문에 답변을 생성하는 API입니다.
    """
    data = request.get_json()

    # 입력 검증
    if not data or 'references' not in data or 'userQuery' not in data:
        return jsonify({"error": "Invalid input format."}), 400

    references = data['references']
    user_query = data['userQuery']

    if len(references) != 3:
        return jsonify({"error": "Exactly 3 reference documents must be provided."}), 400

    # 문서 요약
    summarized_docs = []
    for ref in references:
        title = ref.get('title')
        content = ref.get('content')

        if not title or not content:
            return jsonify({"error": "Each reference must have a title and content."}), 400

        summary = summary_service.summarize_content(content, summary_prompt)
        summarized_docs.append(f"# {title}\n{summary}")

    # 요약된 문서를 기반으로 답변 생성
    combined_summary = "\n\n".join(summarized_docs)
    answer = document_service.answer_question_with_summary(
        combined_summary, 
        user_query, 
        answer_prompt
    )

    return jsonify({
        "ragResponse": answer
    }), 200
