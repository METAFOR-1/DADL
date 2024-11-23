
# 모듈 임포트
import chardet
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn

# 파일 경로 설정
file_path = "/app/muscle_final.csv"

# 파일 인코딩 감지
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# 감지된 인코딩으로 CSV 파일 읽기
df = pd.read_csv(file_path, encoding=encoding)

# 통일할 근육명 정의
replace_dict = {
    '극상근': '극하근',
    '경반극근': '경반극근',
    '소흉근': '소흉근',
    '소원근': '소원근',
    '내측익상근': '내측익하근',
    '외측익상근': '외측익하근',
    '중간광근': '중간광근',
    '대퇴이두근': '대퇴이두근',
    '이두박근': '이두박근',
    '삼두박근,내측두': '삼두박근,외측두',
    '삼두박근,외측두': '삼두박근,외측두',
}

# 근육명 통일
df['근육명_대분류'] = df['근육명'].replace(replace_dict)

# '처한상황과 증상명' 생성
df['처한상황과 증상명'] = df['증상명'].astype(str) + ', ' + df['원인내용'].astype(str)

# Sentence-BERT 모델 로드 (전역에서 한 번만 로드)
model = SentenceTransformer('all-MiniLM-L6-v2')

# BM25 인덱스 생성 (전역에서 한 번만 생성)
corpus = df['처한상황과 증상명'].tolist()
tokenized_corpus = [sentence.split() for sentence in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# FastAPI 애플리케이션 생성
app = FastAPI()

# 요청 데이터 모델 정의
class UserInput(BaseModel):
    user_input: str

@app.post("/predict/")
async def predict(data: UserInput):
    user_input = data.user_input
    if not user_input:
        return {"error": "user_input is required"}

    # 사용자 입력 토큰화
    tokenized_query = user_input.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # BM25 상위 10개 선택
    top_10_indices = np.argsort(bm25_scores)[-10:][::-1]

    # 최종 상위 결과를 찾기 위한 리스트 초기화
    final_results = []

    # 사용자 입력 임베딩 생성
    user_embedding = model.encode([user_input])

    # 상위 10개의 문장 각각에 대해 ',' 단위로 나누어 유사도 계산
    for index in top_10_indices:
        situation_text = df.loc[index, '처한상황과 증상명']
        split_sentences = situation_text.split(',')  # ',' 단위로 문장 나누기

        # 각 문장에 대해 임베딩 및 코사인 유사도 계산
        sentence_embeddings = model.encode(split_sentences)
        similarities = cosine_similarity(user_embedding, sentence_embeddings).flatten()

        # 평균 유사도 계산
        mean_similarity = np.mean(similarities)

        # 최종 결과에 추가
        final_results.append((index, mean_similarity))

    # 유사도가 높은 순으로 정렬
    final_results = sorted(final_results, key=lambda x: x[1], reverse=True)

    # 최종 상위 1개 결과 출력 (근육명_대분류)
    if final_results:
        index = final_results[0][0]  # 상위 1개 선택
        return {"근육명_대분류": df.loc[index, '근육명_대분류']}
    else:
        return {"결과": "없습니다."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run에서 PORT 환경 변수 사용
    uvicorn.run("main:app", host="0.0.0.0", port=port)