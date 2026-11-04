from ollama import chat
from langdetect import detect, DetectorFactory
import pandas as pd
import sqlite3
import re

# 언어 감지 결과 고정을 위한 설정
DetectorFactory.seed = 0

user_question = "hello"

# 데이터베이스 연결
chat_database = sqlite3.connect("chat_database.db")

def divide_info(chat_Log_df, nw):
    # 정규표현식 수정: [] 대신 (?:)를 사용하여 다중 문자열 OR 처리
    when_ko = r"([0-9]+(?:시|분|초|에))"
    where_ko = r"([가-힣]+(?:에서|으로|까지))"
    what_ko = r"([가-힣]+(?:을|를|이|가))"

    when_en = r"([0-9]+(?:\s*at|\s*on|\s*in))" 
    where_en = r"((?:on|at)\s+the\s+\w+)"
    what_en = r"(do\s+\w+)"

    # 빈 DataFrame 병합 시 발생하는 FutureWarning 방지를 위해 nw를 DataFrame으로 만들 때 값이 없으면 빈 리스트 대신 None 사용
    df = chat_Log_df.copy()
    nw_df = pd.DataFrame([nw])
    df = pd.concat([df, nw_df], ignore_index=True)

    if nw['lang'] == 'ko':
        # expand=False를 통해 Series 형태로 반환받음
        df['when'] = df['content'].str.extract(when_ko, expand=False)
        df['where'] = df['content'].str.findall(where_ko)
        df['what'] = df['content'].str.findall(what_ko)

    elif nw['lang'] == 'en':
        df['when'] = df['content'].str.extract(when_en, expand=False)
        df['where'] = df['content'].str.findall(where_en)
        df['what'] = df['content'].str.findall(what_en)

    else:
        print("아직 구현 못함")
        df['when'] = None
        df['where'] = None
        df['what'] = None

    # [핵심 수정] DB에는 리스트를 넣을 수 없으므로, 리스트 요소들을 콤마(,)로 연결된 문자열로 변환
    df['where'] = df['where'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['what'] = df['what'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    return df

# 초기 데이터프레임 구조 정의 (경고 방지)
chat_log = pd.DataFrame(columns=['role', 'lang', 'content', 'when', 'where', 'what'])

# 1. 사용자 질문 처리
try: 
    user_language = detect(user_question)
except:
    user_language = "unknown"

minilog_user = {'role': 'user', 'lang': user_language, 'content': user_question}
chat_log = divide_info(chat_log, minilog_user)

# 2. 모델 응답 생성
response = chat(
    model='smollm',
    messages=[{'role': 'user', 'content': user_question}],
)
ollama_answer = response['message']['content']

# 3. 모델 응답 처리
try: 
    ollama_language = detect(ollama_answer)
except:
    ollama_language = "unknown"

# role을 'user'에서 'agent'로 수정
minilog_agent = {'role': 'agent', 'lang': ollama_language, 'content': ollama_answer}
chat_log = divide_info(chat_log, minilog_agent)

# 4. DB에 저장 (오류 해결됨)
chat_log.to_sql("chat_logs", chat_database, if_exists="append", index=False)
chat_database.close()

print(chat_log)