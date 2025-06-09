import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, LANGCHAIN_MODEL, CATEGORY
from utils.error_handler import handle_error

# 1. LLM 인스턴스 생성. temperature를 낮춰서 답변 생성
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0.1,
)

# 2. JSON Schema 정의
json_schema = {
    "title": "extract_metadata",
    "description": "Extract keywords and classify the chat as DEV_DOC or MEETING_DOC",
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "핵심 키워드 리스트",
        },
        "category": {
            "type": "string",
            "enum": [CATEGORY.DEV_DOC, CATEGORY.MEETING_DOC],
            "description": "문서 유형",
        },
    },
    "required": ["keywords", "category"],
}

# 3. 구조화된 출력 LLM 생성
structured_llm = llm.with_structured_output(schema=json_schema)

# 4. Prompt 정의
prompt_template = PromptTemplate(
    input_variables=["chat_context"],
    template=(
        "아래 채팅 로그를 분석하여 핵심 키워드를 뽑고,\n"
        "개발 문서면 DEV_DOC, 회의록이면 MEETING_DOC으로 분류해주세요.\n"
        "JSON 형식으로만 결과를 반환해주세요.\n\n"
        "채팅 로그:\n{chat_context}"
    )
)

# 5. 함수 정의
def extract_keywords_and_category(chat_context: str) -> dict:
    """채팅 로그를 분석하여 핵심 키워드와 카테고리를 추출"""
    formatted_prompt = prompt_template.format(chat_context=chat_context)

    try:
        result = structured_llm.invoke(formatted_prompt)
        return result
    except Exception as e:
        try:
            fallback_result = {
                "keywords": [],
                "category": CATEGORY.DEV_DOC
            }
            return fallback_result
        except Exception:
            return handle_error(
            "Error extracting keywords and category",
            "키워드 및 카테고리 추출에 실패했습니다.",
            500
        )
