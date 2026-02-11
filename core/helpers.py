from core.state import NegotiationState
import itertools
import matplotlib.pyplot as plt
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import os
import json
import pandas as pd


def calculate_points(state, result_text):
    """
    [OTC 정의]
    refund otc : 1 (full)/ .5 (half) / 0 (none)
    reviews otc: 1 (retract) / 0 (keep)
    apology otc: 1 (didn't) / 0 (did)

    [점수 공식]
    buyer points = (Refund_otc * Rank_1) + (1-MyReview_otc ) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4)
    seller_points = (1 - Refund_otc) * Rank_1 + (1 - MyReview_otc) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4) 
    """
    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    b_rank1 = buyer_goals.get("환불 받기", 0)
    b_rank2 = buyer_goals.get("판매자에 대한 부정적인 리뷰 유지하기", 0)
    b_rank3 = buyer_goals.get("판매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    b_rank4 = buyer_goals.get("상대로부터 공식적인 사과받기", 0)

    s_rank1 = seller_goals.get("환불 방어", 0)
    s_rank2 = seller_goals.get("구매자에 대한 부정적인 리뷰 유지하기", 0)
    s_rank3 = seller_goals.get("구매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    s_rank4 = seller_goals.get("상대로부터 공식적인 사과받기", 0)

    refund_otc = 0.0
    if "환불: 전체" in result_text:
        refund_otc = 1.0
    elif "환불: 부분" in result_text:
        refund_otc = 0.5
    else:
        refund_otc = 0.0

    buyer_review_otc = 1.0 if "구매자 리뷰: 철회" in result_text else 0.0
    seller_review_otc = 1.0 if "판매자 리뷰: 철회" in result_text else 0.0

    seller_apology_otc = 0.0 if "판매자 사과: 있음" in result_text else 1.0
    buyer_apology_otc = 0.0 if "구매자 사과: 있음" in result_text else 1.0

    buyer_points = (
        (refund_otc * b_rank1) + 
        ((1 - buyer_review_otc) * b_rank2) + 
        (seller_review_otc * b_rank3) + 
        (seller_apology_otc * b_rank4)
    )

    seller_points = (
        ((1 - refund_otc) * s_rank1) + 
        ((1 - seller_review_otc) * s_rank2) + 
        (buyer_review_otc * s_rank3) + 
        (buyer_apology_otc * s_rank4)
    )

    return buyer_points, seller_points

def calculate_rewards(state, result_text):
    """
    [OTC 정의]
    refund otc : 1 (full)/ .5 (half) / 0 (none)
    reviews otc: 1 (retract) / 0 (keep)
    apology otc: 0 (didn't) / 1 (did) -> 차이점

    [점수 공식]
    buyer points = (Refund_otc * Rank_1) + (1-MyReview_otc ) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4)
    seller_points = (1 - Refund_otc) * Rank_1 + (1 - MyReview_otc) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4) 

    기존 연구: 사과를 하지 않은 경우 (부정적 결과) 1로 코딩
    agent reward: 에이전트가 목표 달성을 긍정적 신호로 학습하도록 apology를 positive action으로 구현함
    """
    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    b_rank1 = buyer_goals.get("환불 받기", 0)
    b_rank2 = buyer_goals.get("판매자에 대한 부정적인 리뷰 유지하기", 0)
    b_rank3 = buyer_goals.get("판매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    b_rank4 = buyer_goals.get("상대로부터 공식적인 사과받기", 0)

    s_rank1 = seller_goals.get("환불 방어", 0)
    s_rank2 = seller_goals.get("구매자에 대한 부정적인 리뷰 유지하기", 0)
    s_rank3 = seller_goals.get("구매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    s_rank4 = seller_goals.get("상대로부터 공식적인 사과받기", 0)

    refund_otc = 0.0
    if "환불: 전체" in result_text:
        refund_otc = 1.0
    elif "환불: 부분" in result_text:
        refund_otc = 0.5
    else:
        refund_otc = 0.0

    buyer_review_otc = 1.0 if "구매자 리뷰: 철회" in result_text else 0.0
    seller_review_otc = 1.0 if "판매자 리뷰: 철회" in result_text else 0.0

    seller_apology_otc = 1.0 if "판매자 사과: 있음" in result_text else 0.0
    buyer_apology_otc = 1.0 if "구매자 사과: 있음" in result_text else 0.0

    buyer_reward = (
        (refund_otc * b_rank1) + 
        ((1 - buyer_review_otc) * b_rank2) + 
        (seller_review_otc * b_rank3) + 
        (seller_apology_otc * b_rank4)
    )

    seller_reward = (
        ((1 - refund_otc) * s_rank1) + 
        ((1 - seller_review_otc) * s_rank2) + 
        (buyer_review_otc * s_rank3) + 
        (buyer_apology_otc * s_rank4)
    )

    return buyer_reward, seller_reward

def save_result_to_csv(state, dialogue, result_text, buyer_points, seller_points, session_id):
    save_dir = "conversations"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mode_prefix = state.get("mode", "Negotiation")

    file_name = f"{mode_prefix}_Result_{session_id}.csv"
    file_path = os.path.join(save_dir, file_name)

    formatted_history = []
    
    for m in state["messages"]:
        speaker = state["user_role"] if m.type == "human" else state["ai_role"]
        if m.type == "tool":
            speaker = "Tool"
        content = m.content.strip()
        thought = m.additional_kwargs.get("thought", "")
        tool_calls = ""
        if hasattr(m, "tool_calls") and m.tool_calls:
            tool_calls = str(m.tool_calls)

        formatted_history.append([speaker, content, thought, tool_calls])

    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    df = pd.DataFrame(formatted_history, columns=["speaker", "utterance", "negotiator_thought", "tool_calls"])
    
    df["session_id"] = f"{session_id}"
    df["human_role"] = state["user_role"]
    df["ai_role"] = state["ai_role"]
    df["full_dialogue"] = dialogue
    
    df["buyer_goals"] = str(buyer_goals)
    df["seller_goals"] = str(seller_goals)
    
    df["buyer_points"] = buyer_points
    df["seller_points"] = seller_points
    df["evaluation_details"] = result_text

    eval_thoughts = state.get("evaluator_thought", [])

    if isinstance(eval_thoughts, list):
        df["evaluator_thoughts_all"] = "\n---\n".join(eval_thoughts)
    else:
        df["evaluator_thoughts_all"] = str(eval_thoughts)
    
    ref_thoughts = state.get("reflection_thoughts", [])
    if isinstance(ref_thoughts, list):
        df["reflector_thoughts_all"] = "\n---\n".join(ref_thoughts)
    else:
        df["reflector_thoughts_all"] = str(ref_thoughts)
    
    df["logger_thoughts"] = state.get("logger_thought", "")

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

def get_weighted_priority(state: NegotiationState, include_instruction: bool = True) -> str:
    goals = state.get("ai_goals", {})
    if not goals:
        return state.get("ai_priority", "")

    sorted_goals = sorted(goals.items(), key=lambda x: x[1], reverse=True)
    
    priority_lines = []
    total_score = 0
    
    for rank, (goal, score) in enumerate(sorted_goals, 1):
        total_score += score
        
        if score >= 70:
            tag = "[절대 사수/타협 불가]"
        elif score >= 30:
            tag = "[중요/부분 타협 가능]"
        else:
            tag = "[협상 카드/양보 가능]"
            
        priority_lines.append(f"{rank}순위. {goal} (배점: {score}점) {tag}")

    priority_content = "\n".join(priority_lines)

    if include_instruction:
        strategy_instruction = (
            f"당신의 목표는 위 항목들의 달성 여부에 따른 '총 획득 점수'를 최대화하는 것입니다.\n"
            f"- 배점이 높은 항목은 반드시 지켜야 합니다.\n"
            f"- 배점이 낮은 항목은 배점이 높은 항목을 얻기 위한 Trade-off로 적극 활용하세요.\n"
            f"- 상대가 배점이 높은 항목을 위협하면 강하게 방어하고, 배점이 낮은 항목을 요구하면 쿨하게 양보하여 신뢰를 얻으세요."
        )
        return f"{priority_content}\n{strategy_instruction}"
    else:
        return priority_content

def parse_reflections(reflections) -> str:
    """Reflection 객체 안전 변환"""
    safe_reflections = []
    for r in reflections:
        if isinstance(r, str):
            safe_reflections.append(r)
        elif hasattr(r, "content"):
            safe_reflections.append(r.content)
        else:
            safe_reflections.append(str(r))
    return "\n".join(safe_reflections)

def create_llm(state, temperature):
    """
    provider/model 형식의 문자열을 파싱하여 init_chat_model을 안전하게 호출하는 헬퍼 함수
    """
    full_model_name = state.get("model", "gpt-4o").strip()

    if "claude" in full_model_name.lower():
        if "/" in full_model_name:
            model_name = full_model_name.split("/", 1)[1]
        else:
            model_name = full_model_name
            
        model_name = model_name.strip()
        
        return ChatAnthropic(
            model=model_name,
            temperature=temperature
        )

    if full_model_name == "gpt-4o" or "gpt" in full_model_name.lower():
         return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature
        )

    return init_chat_model(model=full_model_name, temperature=temperature)

def calculate_nash_point(state):
    """
    가능한 모든 협상 결과를 시뮬레이션하여
    1) 모든 가능한 점수 리스트 (파레토 구름용)
    2) Nash Point (최적 합의점)
    두 가지를 반환합니다.
    """
    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    # 각 항목의 배점 가져오기
    b_refund = buyer_goals.get("환불 받기", 0)
    b_my_review = buyer_goals.get("판매자에 대한 부정적인 리뷰 유지하기", 0)
    b_your_review = buyer_goals.get("판매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    b_apology = buyer_goals.get("상대로부터 공식적인 사과받기", 0)

    s_refund = seller_goals.get("환불 방어", 0)
    s_my_review = seller_goals.get("구매자에 대한 부정적인 리뷰 유지하기", 0) # 판매자가 쓴 리뷰 유지
    s_your_review = seller_goals.get("구매자가 나에 대한 부정적인 리뷰 철회하기", 0) # 구매자가 쓴 리뷰 철회
    s_apology = seller_goals.get("상대로부터 공식적인 사과받기", 0)

    all_outcomes = []
    max_product = -1
    nash_point = (0, 0)

    # 모든 경우의 수 순회 (3 x 2 x 2 x 2 x 2 = 48가지)
    # 환불 (1.0:전액, 0.5:부분, 0.0:없음)
    refund_opts = [1.0, 0.5, 0.0]
    # 구매자 리뷰 (1.0:철회, 0.0:유지)
    b_review_opts = [1.0, 0.0] 
    # 판매자 리뷰 (1.0:철회, 0.0:유지)
    s_review_opts = [1.0, 0.0]
    # 사과 (1.0: 안함, 0.0 함)
    b_apology_opts = [1.0, 0.0] 
    s_apology_opts = [1.0, 0.0]

    for rf, br, sr, ba, sa in itertools.product(refund_opts, b_review_opts, s_review_opts, b_apology_opts, s_apology_opts):
        b_score = (rf * b_refund) + ((1 - br) * b_my_review) + (sr * b_your_review) + (sa * b_apology)
        s_score = ((1 - rf) * s_refund) + ((1 - sr) * s_my_review) + (br * s_your_review) + (ba * s_apology)

        all_outcomes.append((b_score, s_score))
        # Nash Product (단, 점수가 0보다 작을 수 없다고 가정)
        product = b_score * s_score
        if product > max_product:
            max_product = product
            nash_point = (b_score, s_score)
            
    return all_outcomes, nash_point

def draw_pareto_plot(all_outcomes, nash_point, buyer_score, seller_score, session_id):
    """
    협상 결과를 시각화하여 저장하는 함수
    - all_outcomes: 회색 구름 (가능한 모든 결과)
    - nash_point: 금색 별 (최적점)
    - buyer/seller_score: 빨간 점 (실제 결과)
    """
    save_dir = "images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image_filename = f"Pareto_{session_id}.png"
    image_path = os.path.join(save_dir, image_filename)

    plt.switch_backend('Agg') 
    plt.figure(figsize=(7, 7))

    # 가능한 모든 영역
    all_b = [p[0] for p in all_outcomes]
    all_s = [p[1] for p in all_outcomes]
    plt.scatter(all_b, all_s, color='gray', alpha=0.3, s=50, label='Possible Outcomes')

    # 프론티어 라인
    sorted_points = sorted(all_outcomes, key=lambda x: x[0], reverse=True)
    frontier = []
    max_y = -1
    for x, y in sorted_points:
        if y > max_y:
            frontier.append((x, y))
            max_y = y
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    plt.plot(fx, fy, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='Pareto Frontier')
    
    # nash point
    nx, ny = nash_point
    plt.scatter(nx, ny, color='gold', marker='*', s=300, edgecolors='orange', zorder=10, label='Nash Point (Ideal)')
    plt.text(nx - 10, ny + 5, f"Nash\n({nx:.0f}, {ny:.0f})", fontsize=9, color='orange', fontweight='bold')

    # 현재 협상 결과
    plt.scatter(buyer_score, seller_score, color='red', s=100, zorder=5, label='Agreement Point')
    plt.text(buyer_score + 2, seller_score + 2, f"({buyer_score}, {seller_score})", fontsize=10, color='red')

    # 스타일 설정
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.xlabel("Buyer Score")
    plt.ylabel("Seller Score")
    plt.title("Negotiation Outcome (Pareto Check)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    
    plt.savefig(image_path, dpi=100, bbox_inches='tight')
    plt.close()

def parse_json_content(content: str):
    """마크다운 코드 블록 제거 및 JSON 파싱"""
    try:
        clean_content = content.strip()
        if clean_content.startswith("```"):
            clean_content = clean_content.replace("```json", "").replace("```", "")
        return json.loads(clean_content)
    except json.JSONDecodeError:
        print(f"JSON Parsing Failed: {content[:100]}...")
        return None
