from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

def answer_question_with_summary(summary: str, user_query: str, prompt_template: str) -> str:
    """
    요약문과 사용자 질문을 바탕으로 최종 답변을 생성합니다.
    """
    prompt = prompt_template.format(summary=summary, question=user_query)
    response = llm(prompt)
    return response.strip()

def answer_question_without_docs(user_query: str, prompt_template: str) -> str:
    """
    사용자 질문에 대한 답변을 생성합니다.
    """
    prompt = prompt_template.format(question=user_query)
    response = llm(prompt)
    return response.strip()
