import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import tiktoken

load_dotenv(find_dotenv())

API_KEY = os.environ["API_KEY"]
SYSTEM_MESSAGE = os.environ["SYSTEM_MESSAGE"]

BASE_URL = "https://api.together.xyz"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
FILENAME = "message_history.json"
INPUT_TOKEN_LIMIT = 2048 # 내가 설정한 토큰의 최대 개수 제한?-> 메모리와 처리 속도 한계/대부분 2048이면 처리 가능능

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) # 서버한테 요청을 보내고 결과를 받아오는..?



def chat_completion(messages, model=DEFAULT_MODEL, temperature=0.0, **kwargs): # 0.1 -> 정확하고 일관된 답변 받기 위해.-서.
    response = client.chat.completions.create(
        model=model,#위에서 설정한llama모델?
        messages=messages, #이 messages에 대한 코드는 밑에서 설정함.
        temperature=temperature, # 얼마나 다양하게(창의적으로) 표현할지 조절. 그게 위에서 0.1로 설정한거임.
        stream=False, # stream : 흐름, 연속적으로 나오는 데이터 조각들 stream = False 는 답변 전체가 완성되고 보여줌. 
        **kwargs,# 추가 설정 가능
    )
    return response.choices[0].message.content
# completion -> 한 번에 전체 응답을 받음/ API 호출 후 응답이 다 올 때까지 기다렸다가 한꺼번에 내용을 받음(가장 일반적인 방식식)
#            -> 간단한 챗봇, 한 줄 응답

#completion_stream -> 응답을 조각별로 실시간 출력하면서 받아오는 함수/ 답변이 생성되는 즉시 조각조각 받아서 바로바로 출력 가능
#                  -> 콘솔 앱, 웹 앱에서 실시간 출력 

# 두 함수를 꼭 같이 써야하는 건 아니지만 둘 다 만들어 놓으면 상황에 맞게 편리하게 선택 가능함.
def chat_completion_stream(messages, model=DEFAULT_MODEL, temperature=0.0, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True, # stream=True 는 조금씩 받아서 바로바로 보여줌.
        **kwargs,
    )
    
    #stream 응답 처리 
    response_content = ""

    for chunk in response: # response-> openai api로부터 받은 stream 응답 객체임/ chunk(조각)
        chunk_content = chunk.choices[0].delta.content # chunk 하나에 모델이 생성한 텍스트의 일부가 들어가 있음. / 0 은 답변 후보 중 첫 번째 것을 선택한다는 뜻임./ .delta.content: 이번에 생성된 조각에서 받은 텍스트 내용용
        if chunk_content is not None: # 응답에서 delta.content가 없을 수 있으니 None 체크 해야함.
            print(chunk_content, end="") # 바로바로 출력 후 실시간으로 보여주기 위함. end="" -> 줄바꿈 X
            response_content += chunk_content #실시간 출력 동시에 전체 답변을 문자열로 저장. -> 리턴해서 챗봇의 메시지 기록에 쓸 수 있음 

    print()
    return response_content


# 모델에 보낼 메시지의 토큰 수 계산, 많을 경우 잘라내는 역할 

# text 문자열을 토큰으로 바꾸고 토큰의 개수 리턴.
#LLM은 단어 단위 X 토큰 단위로 문장 처리.
def count_tokens(text, model): #  text : 토큰 수 세고 싶은 문자열 받는 매개변수임/ 문자열이 몇 개의 토큰으로 이루어졌는지 반환
    encoding = tiktoken.get_encoding("cl100k_base")  #encoding : 텍스트->토큰 (텍스트는 바로 모델로 들어갈 수 없기때뮨) / 토크나이저 : openai에서 만든 라이브러리/ cl100k_base : 기본 토큰 인코딩 방식식
    tokens = encoding.encode(text)#text를 토큰 리스트로 변환환
    return len(tokens)#토큰 개수 반환환

#messages리스트에 있는 모든 메시지들의 토큰 수 더해서 총합을 계산.
def count_total_tokens(messages, model):
    total = 0 # 토큰 개수 누적할 변수. 처음에는 0에서 시작, 하나씩 더ㅐ함함
    for message in messages: 
        total += count_tokens(message["content"], model) # count_tokens함수에 넣어서 토큰 수 계산함 그리고 그 값을 total에 계속 누적함함
    return total

#전체 토큰 수가 token_limit을 초과하면 가장 오래된 메시지부터 제거함.
def enforce_token_limit(messages, token_limit, model=DEFAULT_MODEL): # 지금까지 대화 목록, 허용된 최대 토큰수(위에서 설정한거), 사용할모델이름름
    while count_total_tokens(messages, model) > token_limit:# 전체 메시지 합친 토큰 수 > token_limit ->반복 / 너무 많으면 자르기 시작작
        if len(messages) > 1:# 1초과->제거거
            messages.pop(1)#두 번쨰 메시지 삭제 
        else:
            break # 메시지 1개 이하 -> 종료
        # 입력 토큰 제항이 있어서 자동으로 맞춰주는 로직이 필요한가가

#json파일관리함수
def save_to_json_file(obj, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4, ensure_ascii=False)

def load_from_json_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"{filename} 파일을 읽는 중 오류 발생: {e}")
        return None
    
#챗봇 메인 함수수
def chatbot():
    messages = load_from_json_file(FILENAME)
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    print("Chatbot: 안녕하세요! 무엇을 도와드릴까요? (종료하려면 'quit' 또는 'exit'을 입력하세요.)")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        messages.append({"role": "user", "content": user_input})

        total_tokens = count_total_tokens(messages, DEFAULT_MODEL)
        print(f"[현재 토큰 수: {total_tokens} / {INPUT_TOKEN_LIMIT}]")

        enforce_token_limit(messages, INPUT_TOKEN_LIMIT)

        print("Chatbot: ", end="")
        response = chat_completion_stream(messages)
        print()

        messages.append({"role": "assistant", "content": response})

        save_to_json_file(messages, FILENAME)

chatbot()
