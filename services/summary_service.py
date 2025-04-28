from langchain.llms import OpenAI

# LLM 초기화 (Production에서는 별도 config 관리 권장)
llm = OpenAI(temperature=0)

def summarize_content(content: str, prompt_template: str) -> str:
    """
    Robust 프롬프트 기반 문서 요약
    """
    prompt = prompt_template.format(content=content.strip())
    result = llm(prompt)
    return result.strip()

