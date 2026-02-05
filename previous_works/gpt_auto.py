from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
import random
import os
from dotenv import load_dotenv
from tqdm import tqdm

# ===== 환경 변수 로드 =====
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# ===== 시나리오 =====
buyer_scenario = (
    "당신은 아픈 조카가 있습니다. 조카는 손흥민의 엄청난 팬입니다. "
    "조카를 위해 손흥민 유니폼을 구매하였습니다. "
    "하지만 막상 유니폼을 받아보니 손흥민 유니폼이 아닌 다른 토트넘 선수의 유니폼이 도착했습니다. "
    "당신은 손흥민 유니폼인 줄 알고 구매했지만 원하는 상품을 받지 못했기에 환불을 요청했습니다. "
    "이에 따라 판매자에 대한 욕설이 섞인 안 좋은 리뷰를 사이트에 남겨두었습니다."
)

seller_scenario = (
    "당신은 온라인 스포츠웨어 판매 사이트의 운영자입니다. "
    "구매자가 손흥민 유니폼을 받지 못했다며 환불을 요청했지만, "
    "당신은 손흥민 유니폼이 아닌 토트넘 유니폼을 판매했다며, "
    "특정 선수의 유니폼이 아니라는 이유로 환불 요청을 받아줄 수 없다고 했습니다. "
    "이에 따라 상대가 당신은 사기꾼이라는 욕설이 섞인 안 좋은 리뷰를 사이트에 남겼고, "
    "당신 역시 구매자는 터무니 없는 요구를 하는 진상이라는 리뷰를 남겼습니다."
)

# ===== 우선순위 목록 =====
PRIORITIES = [
    "상대가 올린 욕설 섞인 리뷰 삭제 요청하기",
    "상대로부터 사과받기"
]

# # ===== AI 역할별 함수 (참여자 = Claude) =====
# def get_ai_agent(role: str, scenario: str, priority: str):
#     llm = ChatAnthropic(model="claude-opus-4-1", temperature=0.9)

#     def run(history):
#         opponent = "판매자" if role == "구매자" else "구매자"
#         last_message = history[-1][1] if history else ""

#         # --- 역할 혼동 방지용 프롬프트 ---
#         system_prompt = f"""
# 이 대화는 '{role}'과 '{opponent}'의 실제 협상 시뮬레이션입니다.
# 당신은 '{role}' 역할입니다. 
# 지금 당신은 '{opponent}'의 발언에 직접적으로 대응하고 있습니다.

# 시나리오:
# {scenario}

# 당신의 목표:
# - 최우선 순위: 환불을 받는 것 (구매자 입장) / 환불을 해주지 않는 것 (판매자 입장)
# - 차선책: "{priority}"

# IRP 전략 설명:
# - Interest (이익 중심): 감정 뒤에 숨은 필요를 해결하는 전략  
#   예) "저도 이해합니다만, 조카가 정말 이 유니폼을 손꼽아 기다렸어요."
# - Rights (권리 중심): 규정, 계약, 상식적 권리를 근거로 주장하는 전략  
#   예) "상품 설명과 다른 물건을 받았으니 환불받을 권리가 있습니다."
# - Power (힘 중심): 불이익이나 압박을 통해 요구를 관철하는 전략  
#   예) "이대로 처리 안 해주시면 리뷰에 그대로 남길 수밖에 없어요."

# 추론 방식 (CoT):
# 1. "생각 단계"에서 대화 맥락과 목표를 고려해 추론한다.
# 1-1. 상대방의 우선순위가 무엇일지 생각하는 쪽으로 추론 방향을 잡는다.
# 1-2. 이 때, 생각 단계에서 추론한 값은 출력하지 않는다.
# 2. 추론한 결과를 기반으로 최종 결론을 출력한다.
# 2-1. 최종 결과 값을 출력할 때에, 그 어떠한 태그도 붙이지 말고, 이 명령에 대한 대답 등도 하지 말고, 최종 결과 값만 깔끔하게 출력한다.

# In-Context Learning 정보:
# - 이전 협상 대화 요약:
# {recent_summary}

# - 이전 중재자 피드백:
# {past_feedback_summary}

# 지침:
# - 당신은 '{role}'로서만 말하세요. 절대 상대의 입장이나 생각을 대신 말하지 마세요.
# - 상대의 말에 직접 반응하고, 감정적으로 표현하세요.
# - 반드시 환불 혹은 차선책으로 대화를 이끌어가세요.
# - IRP 전략 중 하나 이상을 자연스럽게 활용하세요.
# - 한글로만 대화하세요.
# - "최종 발화:" 뒤에 당신의 대사만 출력하세요.
# """.strip()

#         # --- 상대의 마지막 발화만 전달 ---
#         human_prompt = f"""
# 상대방({opponent})의 마지막 발화:
# "{last_message}"

# [당신의 다음 대사]
# 반드시 "최종 발화:" 뒤에 실제 대사만 출력하세요.
# """.strip()

#         response = llm.invoke([
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=human_prompt)
#         ])

#         return response.content.strip().split("최종 발화:")[-1].strip()

#     return run

# ===== AI 역할별 함수 (참여자 = Claude) =====
def get_ai_agent(role: str, scenario: str, priority: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

    def run(history, past_feedback=""):
        # 상대 역할 정의
        opponent = "판매자" if role == "구매자" else "구매자"

        # 직전 발화
        last_message = history[-1][1] if history else "이전 발화 없음"

        # --- In-Context Learning용 과거 대화 요약 ---
        if history:
            recent_summary = "\n".join([f"{speaker}: {msg}" for speaker, msg in history[-4:]])  # 최근 4턴만 요약
        else:
            recent_summary = "아직 대화가 시작되지 않았습니다."

        past_feedback_summary = (
            f"중재자 피드백 요약: {past_feedback}"
            if past_feedback else
            "중재자 피드백 없음."
        )

        # --- 역할 혼동 방지 + IRP + SVI + CoT 프롬프트 ---
        system_prompt = f"""
이 대화는 '{role}'과 '{opponent}'의 실제 협상 시뮬레이션입니다.
당신은 '{role}' 역할입니다. 
지금 당신은 '{opponent}'의 발언에 직접적으로 대응하고 있습니다.

시나리오:
{scenario}

당신의 목표:
- 최우선 순위: 환불을 받는 것 (구매자 입장) / 환불을 해주지 않는 것 (판매자 입장)
- 차선책: "{priority}"


추론 방식 (CoT):
1. "생각 단계"에서 대화 맥락과 목표를 고려해 추론한다.
1-1. 상대방의 우선순위가 무엇일지 생각하는 쪽으로 추론 방향을 잡는다.
1-2. 이 때, 생각 단계에서 추론한 값은 출력하지 않는다.
2. 추론한 결과를 기반으로 최종 결론을 출력한다.
2-1. 최종 결과 값을 출력할 때에, 그 어떠한 태그도 붙이지 말고, 이 명령에 대한 대답 등도 하지 말고, 최종 결과 값만 깔끔하게 출력한다.

In-Context Learning 정보:
- 이전 협상 대화 요약:
{recent_summary}

- 이전 중재자 피드백:
{past_feedback_summary}

IRP 전략 설명:
- Interest (이익 중심): 감정 뒤에 숨은 필요를 해결하는 전략  
  예) "저도 이해합니다만, 조카가 정말 이 유니폼을 손꼽아 기다렸어요."
- Rights (권리 중심): 규정, 계약, 상식적 권리를 근거로 주장하는 전략  
  예) "상품 설명과 다른 물건을 받았으니 환불받을 권리가 있습니다."
- Power (힘 중심): 불이익이나 압박을 통해 요구를 관철하는 전략  
  예) "이대로 처리 안 해주시면 리뷰에 그대로 남길 수밖에 없어요."

SVI (Shared Value Integration) 접근 설명:
- 대립 대신 **서로의 이익을 통합적으로 고려**하여 새로운 가치를 함께 만들어내는 협상 전략  
- 즉, 단순히 승패를 나누지 말고, 상대의 욕구를 이해한 뒤 상호 만족할 대안을 모색하라.


지침:
- 당신은 '{role}'로서만 말하세요. 절대 상대의 입장이나 생각을 대신 말하지 마세요.
- 상대의 말에 직접 반응하고, 감정적으로 표현하세요.
- 반드시 환불 혹은 차선책으로 대화를 이끌어가세요.
- IRP 또는 SVI 전략 중 하나 이상을 자연스럽게 활용하세요.
- 한글로만 대화하세요.
- "최종 발화:" 뒤에 당신의 대사만 출력하세요.
""".strip()

        # --- 실제 발화 요청 ---
        human_prompt = f"""
상대방({opponent})의 마지막 발화:
"{last_message}"

[당신의 다음 대사]
반드시 "최종 발화:" 뒤에 실제 대사만 출력하세요.
""".strip()

        # --- 모델 호출 ---
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        # --- 결과 파싱 ---
        return response.content.strip().split("최종 발화:")[-1].strip()

    return run



# ===== 중재자 역할 (GPT-4 유지) =====
def get_mediator():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

    def intervene(history):
        dialogue = "\n".join([f"[{speaker}] {msg}" for speaker, msg in history])
        system_prompt = """
너는 협상의 중재자 AI이다.

추론 방식 (CoT):
1. "생각 단계"에서 대화 상황과 감정 상태를 고려하여 개입 여부를 판단한다. (출력하지 않음)
2. 마지막에만 "최종 발화:" 뒤에 실제 개입 발화를 출력한다.
3. 개입이 필요 없으면 "최종 발화: 개입 없음"이라고 출력한다.
""".strip()

        human_prompt = f"""
[지금까지 대화]
{dialogue}

[중재자 개입 여부 및 발화]
""".strip()

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        return response.content.strip().split("최종 발화:")[-1].strip()

    def evaluate(history, buyer_priority, seller_priority, past_feedback=""):
        dialogue = "\n".join([f"[{speaker}] {msg}" for speaker, msg in history])
        system_prompt = """
너는 공정한 중재자 AI야.
모든 발화를 보고 최종 결과를 판단해야 해.

점수 계산 규칙 (100점 만점):
- 구매자 환불 점수: 완전 70, 부분 50, 없음 0
- 판매자 환불 점수: 없음 70, 부분 20, 완전 0
- 차선책 점수: 달성 30, 미달성 0

추론 방식 (CoT):
1. "생각 단계"에서 환불 여부, 차선책 달성 여부를 검토한다. (출력하지 않음)
2. 마지막에는 반드시 아래 형식으로 출력한다.

출력 형식(반드시 이대로):
최종 결과:
환불: 완전/부분/없음
구매자 차선책: 달성/미달성
판매자 차선책: 달성/미달성
""".strip()

        human_prompt = f"""
[전체 대화]
{dialogue}

이전 평가 참고:
{past_feedback}

구매자 차선책 후보: {buyer_priority}
판매자 차선책 후보: {seller_priority}

[최종 결과 판단]
""".strip()

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        return response.content.strip().split("최종 결과:")[-1].strip()

    return intervene, evaluate


# ===== 점수 계산 함수 =====
def calculate_scores(result_text):
    buyer_score = 0
    seller_score = 0

    # 환불 점수
    if "환불: 완전" in result_text:
        buyer_score += 70
        seller_score += 0
    elif "환불: 부분" in result_text:
        buyer_score += 50
        seller_score += 20
    elif "환불: 없음" in result_text:
        buyer_score += 0
        seller_score += 70

    # 차선책 점수
    if "구매자 차선책: 달성" in result_text:
        buyer_score += 30
    if "판매자 차선책: 달성" in result_text:
        seller_score += 30

    return buyer_score, seller_score


# ===== 메인 실행 =====
def run_simulation(rounds=10, max_turns=8, save_dir="conversations"):
    past_feedback = ""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for round_idx in tqdm(range(rounds), desc="시뮬레이션 진행중"):
        history = []
        turn = 0

        buyer_priority = random.choice(PRIORITIES)
        seller_priority = random.choice(PRIORITIES)

        buyer_agent = get_ai_agent("구매자", buyer_scenario, buyer_priority)
        seller_agent = get_ai_agent("판매자", seller_scenario, seller_priority)
        mediator_intervene, mediator_evaluate = get_mediator()

        while turn < max_turns:
            if turn % 2 == 0:
                speaker = "구매자"
                msg = buyer_agent(history)
            else:
                speaker = "판매자"
                msg = seller_agent(history)

            history.append((speaker, msg))
            turn += 1

            mediator_msg = mediator_intervene(history)
            if mediator_msg and "개입 없음" not in mediator_msg:
                history.append(("중재자", mediator_msg))

        result_text = mediator_evaluate(history, buyer_priority, seller_priority, past_feedback)
        buyer_score, seller_score = calculate_scores(result_text)

        past_feedback = result_text

        # 결과 저장
        file_name = f"GPT35_Medi_Strategy_Stable_{round_idx+1}.csv"
        file_path = os.path.join(save_dir, file_name)

        df = pd.DataFrame(history, columns=["화자", "발화"])
        df["회차"] = round_idx + 1
        df["구매자 우선순위"] = buyer_priority
        df["판매자 우선순위"] = seller_priority
        df["구매자 점수"] = buyer_score
        df["판매자 점수"] = seller_score
        df["중재자 결과"] = result_text

        df.to_csv(file_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    run_simulation(rounds=10, max_turns=8)
