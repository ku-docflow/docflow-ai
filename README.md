# docflow-ai

## Prerequisites

1. Conda 설치

   - [Conda 공식 웹사이트](https://www.anaconda.com/download)에서 anaconda 설치
   - 설치 후 Conda가 제대로 설치되었는지 확인:
     ```bash
     conda --version
     ```

2. 파이썬 가상환경 생성

   - `docflow` Conda 환경을 생성하고 활성화:
     ```bash
     conda create -n docflow python=3.12.2
     conda activate docflow
     ```

3. Docker 설치
   - [Docker 공식 웹사이트](https://www.docker.com/)에서 Docker를 다운로드하고 설치
   - 설치 후 Docker가 제대로 설치되었는지 확인:
     ```bash
     docker --version
     ```
4. Qdrant 설치 및 실행
   - Docker를 사용하여 Qdrant 설치 및 실행:
     ```bash
     docker pull qdrant/qdrant
     ```
   - Docker run - Qdrant DB 기본 포트는 6333
     ```
     docker run -p 6333:6333 \
       -v $(pwd)/path/to/data:/qdrant/storage \
       qdrant/qdrant
     ```
     - 중요 : 해당 부분 `$(pwd)/path/to/data` 수정해서 로컬 DB path를 직접 설정해야 함

## Installation

1. requirements.txt 설치 (중요 : Conda 가상환경이 활성화된 상태에서 설치)

   - `requirements.txt` 파일을 사용하여 필요한 Python 패키지를 설치:
     ```bash
      pip3 install requirements.txt
     ```

2. 환경 변수 설정

   - 프로젝트 루트 디렉토리에 `.env.example` 파일 내용을 토대로 `.env` 파일 생성
     ```
     OPENAI_API_KEY=your_openai_api_key
     LANGCHAIN_MODEL=your_model_name
     PORT=your_flask_port
     QDRANT_HOST=localhost
     QDRANT_PORT=6333
     QDRANT_COLLECTION_NAME=your_collection_name
     QDRANT_API_KEY=your_qdrant_api_key
     ```
   - `QDRANT_API_KEY` : [random API key generator](https://generate-random.org/api-key-generator)에서 생성
   - `QDRANT_COLLECTION_NAME` : 유저가 임의로 설정 (defaults to 'documents')
   - `PORT` : Flask 포트번호 설정 (macOS환경에서는 5000번대 포트 사용 불가)
   - `LANGCHAIN_MODEL=gpt-4o-mini-2024-07-18`로 설정하는 것을 권장

3. 서비스 실행
   - Flask 서버 실행
     ```bash
     python3 app.py
     ```
4. Qdrant DB Collection 자동 생성됐는 지 확인
   - 아래 링크에서 collection 생성 여부 확인
   ```
    http://localhost:6333/dashboard#/collections
   ```

## API Testing

### @검색봇

- POST REQUEST : http://localhost:8081/api/search-document

- MOCK DATA :

```
{
    "references": [
        {
            "title": "JWT를 이용한 인증 시스템 설계",
            "content": "# JWT 인증\n\n이 문서는 우리 팀이 선택한 인증 방법인 JWT(Json Web Token)에 대해 설명한다.\n\n## JWT 구조\nJWT는 Header, Payload, Signature로 구성된다.\n\n## 장점\n- 무상태(stateless) 서버 구현 가능\n- 확장성과 유연성이 뛰어남\n\n## 단점\n- 토큰 탈취 시 위험성 존재\n\n## 결론\n우리 시스템은 JWT를 기본 인증 수단으로 사용한다."
        },
        {
            "title": "OAuth2와 Token 기반 인증 비교",
            "content": "# OAuth2와 토큰 인증 비교\n\nOAuth2는 제3자 서비스 접근을 위해 토큰을 발급하는 프로토콜이다.\n\n## OAuth2 특징\n- 복잡한 흐름 (Authorization Code, Implicit 등)\n- Scope를 통한 세밀한 권한 제어\n\n## 토큰 인증 특징\n- 간단한 토큰 발급과 검증\n- 독립적인 시스템 구성 가능\n\n## 결론\nOAuth2는 외부 연동 시 사용하고, 내부 인증은 JWT로 처리하기로 결정했다."
        },
        {
            "title": "팀 회의록 - 인증 기술 선정",
            "content": "# 인증 기술 선정 회의록\n\n참석자: 김팀장, 이개발자, 박디자이너\n\n## 논의 내용\n- 세션 기반 인증 vs 토큰 기반 인증 비교\n- 토큰 기반 인증이 확장성과 유지보수에 유리하다는 점에 모두 동의\n- JWT를 사용하여 클라이언트가 서버 리소스에 접근하는 구조 채택\n\n## 결론\n- 최종적으로 JWT + Refresh Token 방식을 사용하기로 합의\n- OAuth2는 외부 API 연동 시에만 적용"
        }
    ],
    "userQuery": "우리 팀이 합의한 로그인 인증 기술은 뭐야? JWT인지 토큰 방식인지 알려줘"
}
```

### @생성봇

POST REQUEST : http://localhost:8081/api/process-document

MOCK DATA :

```
{
  "documentId": 1,
  "organizationId": 2,
  "chatContext": "김영수: 오늘 API 설계 회의 시작하겠습니다. 전체 아키텍처는 지난번 회의안 기준으로 진행할게요.\n정하나: 네, 로그인 방식은 JWT 기반으로 하기로 했죠? 토큰 만료 시간은 30분으로 설정하는 걸로.\n박준호: 리프레시 토큰도 도입할까요? 보안 이슈를 고려하면 필요한 것 같아요.\n김영수: 맞습니다. 리프레시 토큰 도입하고, Redis에 저장해서 관리하도록 하죠.\n정하나: 사용자 인증은 OAuth2도 지원해야 할까요? 클라이언트 쪽 요청이 있던데요.\n김영수: 우선순위는 아니지만 구조는 유연하게 설계해두는 게 좋을 것 같네요.\n박준호: 엔드포인트 명세는 Swagger로 문서화하죠. 자동화도 고려해서 FastAPI 쓰는 걸 추천합니다.\n정하나: 좋아요. 모델 스키마는 제가 정의해서 공유드릴게요. 의견 있으시면 코멘트 주세요.\n김영수: 다음으로는 에러 핸들링 규칙 정하죠. 에러 코드는 공통 포맷으로 정의해야 할 것 같습니다.\n박준호: HTTP status code 기반 + 커스텀 메시지 구조로 정리하면 깔끔할 것 같네요.\n김영수: 그럼 다음 회의 전까지 각자 역할 정리해서 진행해주세요. 정하나님은 로그인/인증, 박준호님은 에러 핸들링 및 문서화 부탁드립니다.",
  "userId": "user-1923",
  "createdBy": "김영수",
  "createdAt": "2025-04-14T10:32:00+09:00"
}
```
