# Python 3.9 Slim 이미지 사용
FROM python:3.9-slim

# 작업 디렉터리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# muscle_final.csv 파일 복사
COPY muscle_final.csv /app/muscle_final.csv

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]