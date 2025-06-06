from langchain.llms import OpenAI
import logging

logger = logging.getLogger(__name__)

# LLM 초기화 (Production에서는 별도 config 관리 권장)
llm = OpenAI(temperature=0)

def summarize_content(content: str, prompt_template: str) -> str:
    """
    Robust 프롬프트 기반 문서 요약
    """
    try:
        prompt = prompt_template.format(content=content.strip())
        result = llm.invoke(prompt)
        return result.strip()
    except Exception as e:
        logger.error(f"Error summarizing content: {str(e)}")
        raise