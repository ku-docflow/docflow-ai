from langchain.prompts import PromptTemplate
from utils.error_handler import handle_error
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LANGCHAIN_MODEL
import logging


logger = logging.getLogger(__name__)


# document가 있을 경우 temperature를 낮춰서 답변 생성
llm_with_docs = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0.1,
)

# document가 없을 경우 temperature를 높여서 답변 생성
llm_without_docs = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=LANGCHAIN_MODEL,
    temperature=0.5,
)

def answer_question_with_summary(
    summary: str,
    question: str,
    prompt_template: PromptTemplate,
    memory_context: str = ""
) -> str:
    """
    문서 요약 및 사용자 쿼리를 기반으로 답변 생성
    """
    try:
        # Format the prompt with all context
        formatted_prompt = prompt_template.format(
            summary=summary,
            question=question,
            memory_context=memory_context
        )
        
        # Generate response using the LLM with invoke method
        response = llm_with_docs.invoke(formatted_prompt)
        return response.content
        
    except Exception as e:
        logger.exception("Error generating answer with docs")
        return handle_error("Error generating answer with docs", "문서 기반 답변 생성에 실패했습니다.", 500)

def answer_question_without_docs(user_query: str, prompt_template: str) -> str:
    """
    사용자 질문에 대한 답변을 docs 없이 생성
    """
    try:
        prompt = prompt_template.format(question=user_query)
        response = llm_without_docs.invoke(prompt)
        return response.content
    except Exception as e:
        logger.exception("Error generating answer without docs")
        return handle_error("Error generating answer without docs", "일반 답변 생성에 실패했습니다.", 500)
