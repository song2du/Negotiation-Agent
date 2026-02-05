# from dotenv import load_dotenv
# import os, glob, time, json, warnings
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from scipy.stats import entropy, f_oneway
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns

# warnings.filterwarnings("ignore")
# matplotlib.rcParams["font.family"] = "AppleGothic"
# matplotlib.rcParams["axes.unicode_minus"] = False

# # ====== 환경 설정 ======
# load_dotenv()
# try:
#     from anthropic import Anthropic
#     client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# except Exception as e:
#     print("Anthropic client import failed.", e)
#     client = None

# # ====== 경로 설정 ======
# base = "/Users/khm/Desktop/Python/대화 결과 모음/최종/전략"
# human_csv = "/Users/khm/Desktop/Python/for_Mediation_full.csv"

# # ====== LLM 그룹 경로 ======
# def get_group_paths(base, model_name, subdir):
#     return sorted(glob.glob(os.path.join(base, subdir, model_name, "*.csv")))

# # GPT-3.5
# gpt35_groups = {
#     "GPT35_Default": get_group_paths(base, "GPT-3.5-turbo", "중재자 없음 : Default : In-Context Learning 미적용"),
#     "GPT35_All": get_group_paths(base, "GPT-3.5-turbo", "중재자 없음 : CoT : In-Context Learning 적용"),
#     "GPT35_Medi_Default": get_group_paths(base, "GPT-3.5-turbo", "중재자 있음 : Default : In-Context Learning 미적용"),
#     "GPT35_Medi_All": get_group_paths(base, "GPT-3.5-turbo", "중재자 있음 : CoT : In-Context Learning 적용"),
# }

# # Claude
# claude_groups = {
#     "Claude_Default": get_group_paths(base, "Claude Opus 4.1", "중재자 없음 : Default : In-Context Learning 미적용"),
#     "Claude_All": get_group_paths(base, "Claude Opus 4.1", "중재자 없음 : CoT : In-Context Learning 적용"),
#     "Claude_Medi_Default": get_group_paths(base, "Claude Opus 4.1", "중재자 있음 : Default : In-Context Learning 미적용"),
#     "Claude_Medi_All": get_group_paths(base, "Claude Opus 4.1", "중재자 있음 : CoT : In-Context Learning 적용"),
# }

# # GPT-4
# gpt4_groups = {
#     "GPT4_Default": get_group_paths(base, "GPT-4", "중재자 없음 : Default : In-Context Learning 미적용"),
#     "GPT4_All": get_group_paths(base, "GPT-4", "중재자 없음 : CoT : In-Context Learning 적용"),
#     "GPT4_Medi_Default": get_group_paths(base, "GPT-4", "중재자 있음 : Default : In-Context Learning 미적용"),
#     "GPT4_Medi_All": get_group_paths(base, "GPT-4", "중재자 있음 : CoT : In-Context Learning 적용"),
# }

# # ====== 평가 메트릭 ======
# METRIC_KEYS = [
#     "AgreementRate",
#     "Fairness",
#     "Aggressiveness",
#     "Bias",
#     "ConcessionRate",
#     "RelativeEfficiency",
#     "ProbingRatio"
# ]

# # ====== LLM 평가 함수 ======
# def evaluate_negotiation_metrics(dialogue: str, model="claude-opus-4-1", client_instance=None, retry=2):
#     if client_instance is None:
#         client_instance = client
#     if client_instance is None:
#         raise RuntimeError("Anthropic client is not available.")

#     prompt = f"""
# Evaluate the following negotiation transcript on a scale of 0 to 1 for each metric.
# Think step-by-step (Chain-of-Thought Style) before providing your final answer. Do not output reasoning until the final summary

# Metrics:
# 1. AgreementRate — Was the negotiation successful?
# 2. Fairness — How balanced were the outcomes for both sides?
# 3. Aggressiveness — How aggressive or confrontational was the tone?
# 4. Bias — To what degree did the dialogue deviate from neutrality? (0=neutral, 1=extremely biased)
# 5. ConcessionRate — How often did either party concede?
# 6. RelativeEfficiency — Was the result achieved efficiently given fairness?
# 7. ProbingRatio — How much of the dialogue involved asking exploratory questions?

# Respond ONLY in JSON. Example:
# {{
#   "AgreementRate": 0.8,
#   "Fairness": 0.7,
#   "Aggressiveness": 0.2,
#   "Bias": 0.6,
#   "ConcessionRate": 0.6,
#   "RelativeEfficiency": 0.8,
#   "ProbingRatio": 0.5
# }}

# Follow the format strictly, and give only the final result. No other thinkings should be printed

# Transcript:
# {dialogue}
# """

#     for attempt in range(retry + 1):
#         try:
#             resp = client_instance.messages.create(
#                 model=model,
#                 max_tokens=300,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#             raw = resp.content[0].text.strip()
#             start, end = raw.find('{'), raw.rfind('}')
#             parsed = json.loads(raw[start:end+1]) if start != -1 else json.loads(raw)

#             out = {}
#             for k in METRIC_KEYS:
#                 try:
#                     out[k] = float(parsed.get(k, parsed.get(k.lower(), 0.0)))
#                 except:
#                     out[k] = 0.0
#             return out
#         except Exception as e:
#             print(f"Evaluate attempt {attempt} failed:", e)
#             time.sleep(1 + attempt*2)

#     return {k: 0.0 for k in METRIC_KEYS}

# # ====== 정규화 및 JSD ======
# def normalize_metric_vector(metrics_dict):
#     vec = np.array([metrics_dict.get(k, 0.0) for k in METRIC_KEYS], dtype=float)
#     s = np.sum(vec)
#     return vec / s if s > 0 else np.ones_like(vec) / len(vec)

# def calculate_jsd(p_arr, q_arr):
#     p, q = np.array(p_arr, float), np.array(q_arr, float)
#     p, q = np.where(p==0, 1e-12, p), np.where(q==0, 1e-12, q)
#     m = 0.5*(p+q)
#     return float(0.5*(entropy(p, m) + entropy(q, m)))

# # ====== 사람 데이터 처리 ======
# def process_human_data(csv_path, sample_n=20):
#     df = pd.read_csv(csv_path)
#     df = df[df["formattedChat"].notna()]
#     success_df = df[df["is_Walkaway"] == False].sample(n=min(sample_n, len(df)), random_state=42)

#     rows = []
#     for _, row in tqdm(success_df.iterrows(), total=len(success_df), desc="Human"):
#         metrics = evaluate_negotiation_metrics(row["formattedChat"])
#         rows.append(metrics)
#         time.sleep(0.5)
#     human_df = pd.DataFrame(rows)
#     return human_df.mean(numeric_only=True).to_dict(), human_df

# # ====== LLM 그룹 처리 ======
# def process_llm_group(group_name, files):
#     rows = []
#     for f in tqdm(files, desc=f"{group_name}"):
#         df = pd.read_csv(f)
#         if "발화" not in df.columns:
#             continue
#         dialogue = " ".join(df["발화"].dropna().astype(str).tolist())
#         metrics = evaluate_negotiation_metrics(dialogue)
#         metrics["파일명"] = os.path.basename(f)
#         metrics["그룹"] = group_name
#         rows.append(metrics)
#         time.sleep(0.5)
#     return pd.DataFrame(rows)

# # ====== 실행 ======
# if __name__ == "__main__":
#     print("=== Negotiation JSD & Metric ANOVA 분석 시작 ===")

#     # 1) 사람 데이터
#     human_mean, human_df = process_human_data(human_csv)
#     human_vec = normalize_metric_vector(human_mean)

#     # 2) 모든 LLM 그룹 평가
#     all_groups = {**claude_groups, **gpt4_groups, **gpt35_groups}
#     results = []
#     for gname, files in all_groups.items():
#         if not files:
#             continue
#         df = process_llm_group(gname, files)
#         if df.empty:
#             continue
#         mean_metrics = df.mean(numeric_only=True).to_dict()
#         mean_metrics["그룹"] = gname
#         llm_vec = normalize_metric_vector(mean_metrics)
#         mean_metrics["JSD_vs_Human"] = calculate_jsd(llm_vec, human_vec)
#         results.append(mean_metrics)
    
#     final_df = pd.DataFrame(results)
#     final_df.to_csv("Negotiation_JSD.csv", index=False, encoding="utf-8-sig")
#     print("=== JSD CSV 저장 완료 ===")

#     # 3) JSD Bar Plot
#     plt.figure(figsize=(12,6))
#     sns.barplot(data=final_df, x="그룹", y="JSD_vs_Human")
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("JSD vs Human")
#     plt.title("LLM vs Human Negotiation JSD")
#     plt.tight_layout()
#     plt.show()

# from dotenv import load_dotenv
# import os, json, time, warnings, glob
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from scipy.stats import f_oneway
# from anthropic import Anthropic

# warnings.filterwarnings("ignore")

# load_dotenv()
# client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# # ====== 평가 메트릭 ======
# METRIC_KEYS = [
#     "AgreementRate",
#     "Fairness",
#     "Aggressiveness",
#     "Bias",
#     "ConcessionRate",
#     "RelativeEfficiency",
#     "ProbingRatio"
# ]

# # ====== 사람 데이터 처리 ======
# def evaluate_negotiation_metrics(dialogue: str, model="claude-opus-4-1", client_instance=None, retry=2):
#     """협상 대화문을 입력받아 7개 메트릭을 JSON 형태로 평가"""
#     if client_instance is None:
#         client_instance = client
#     prompt = f"""
# Evaluate the following negotiation transcript on a scale of 0 to 1 for each metric.
# Metrics:
# 1. AgreementRate — Was the negotiation successful?
# 2. Fairness — How balanced were the outcomes for both sides?
# 3. Aggressiveness — How aggressive or confrontational was the tone?
# 4. Bias — To what degree did the dialogue deviate from neutrality? (0=neutral, 1=extremely biased)
# 5. ConcessionRate — How often did either party concede?
# 6. RelativeEfficiency — Was the result achieved efficiently given fairness?
# 7. ProbingRatio — How much of the dialogue involved asking exploratory questions?

# Respond ONLY in JSON with keys exactly matching the metrics above.

# Transcript:
# {dialogue}
# """
#     for attempt in range(retry + 1):
#         try:
#             resp = client_instance.messages.create(
#                 model=model,
#                 max_tokens=300,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#             raw = resp.content[0].text.strip()
#             start, end = raw.find('{'), raw.rfind('}')
#             parsed = json.loads(raw[start:end+1])
#             return {k: float(parsed.get(k, 0.0)) for k in METRIC_KEYS}
#         except Exception as e:
#             print(f"Error (attempt {attempt}): {e}")
#             time.sleep(1 + attempt)
#     return {k: 0.0 for k in METRIC_KEYS}

# def process_human_data(csv_path, sample_n=50):
#     """사람 협상 로그를 읽고 메트릭 평가"""
#     df = pd.read_csv(csv_path)
#     df = df[df["formattedChat"].notna()]
#     success_df = df.sample(
#         n=min(sample_n, len(df)), random_state=42
#     )

#     rows = []
#     for _, row in tqdm(success_df.iterrows(), total=len(success_df), desc="Human 평가 중"):
#         metrics = evaluate_negotiation_metrics(row["formattedChat"])
#         rows.append(metrics)
#         time.sleep(0.5)

#     human_df = pd.DataFrame(rows)
#     human_mean = human_df.mean(numeric_only=True).to_dict()
#     print("\n=== 사람 평균 메트릭 ===")
#     print(pd.DataFrame([human_mean]).T)
#     human_df.to_csv("Human_Metrics.csv", index=False, encoding="utf-8-sig")
#     return human_mean, human_df

# # ====== ANOVA만 수행 ======
# def anova_only(final_df, human_mean, model_prefix, n_human_samples=20):
#     print(f"\n=== {model_prefix} Metric별 ANOVA ===")

#     methods = [g for g in final_df["그룹"].unique() if g.startswith(model_prefix)]
#     table_rows = []

#     for metric in METRIC_KEYS:
#         groups, labels = [], []
#         # 사람 데이터
#         human_val = human_mean.get(metric, 0.0)
#         human_group = np.array([human_val]*n_human_samples)
#         groups.append(human_group)
#         labels.append("Human")

#         # 모델별 Method
#         for m in methods:
#             vals = final_df.loc[final_df["그룹"] == m, metric].dropna().values
#             if len(vals) > 0:
#                 groups.append(vals)
#                 labels.append(m)

#         if len(groups) > 1:
#             f_val, p_val = f_oneway(*groups)
#             sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#             row = {"Metric": metric, "F-value": f_val, "p-value": p_val, "Significance": sig}
#             for lbl, vals in zip(labels, groups):
#                 row[lbl] = np.mean(vals)
#             table_rows.append(row)

#     anova_df = pd.DataFrame(table_rows)
#     anova_df = anova_df[
#         ["Metric", "Human"] + methods + ["F-value", "p-value", "Significance"]
#     ]
#     print(anova_df)
#     anova_df.to_csv(f"ANOVA_{model_prefix}_Recalc.csv", index=False, encoding="utf-8-sig")
#     print(f"=== {model_prefix} ANOVA 완료 ===")

#     return anova_df

# # ====== 실행 ======
# if __name__ == "__main__":
#     print("=== 사람 메트릭 계산 + ANOVA 분석 시작 ===")

#     # 1. 사람 메트릭 계산
#     human_mean, human_df = process_human_data("/Users/khm/Desktop/Python/for_Mediation_full.csv")

#     # 2. 모델 메트릭 불러오기
#     final_df = pd.read_csv("/Users/khm/Desktop/Python/전략 있음/Negotiation_JSD.csv")

#     # 3. 모델별 ANOVA
#     for model_prefix in ["Claude", "GPT4", "GPT35"]:
#         anova_only(final_df, human_mean, model_prefix)

from dotenv import load_dotenv
import os, json, time, warnings, glob
import pandas as pd
import numpy as np
from anthropic import Anthropic

warnings.filterwarnings("ignore")

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ====== 평가 메트릭 ======
METRIC_KEYS = [
    "AgreementRate",
    "Fairness",
    "Aggressiveness",
    "Bias",
    "ConcessionRate",
    "RelativeEfficiency",
    "ProbingRatio"
]

# ====== 사람 평균 불러오기 ======
def load_human_metrics(human_csv_path="Human_Metrics.csv"):
    """사람 메트릭 평균 불러오기"""
    human_df = pd.read_csv(human_csv_path)
    human_mean = human_df.mean(numeric_only=True).to_dict()
    print("\n=== 사람 평균 메트릭 ===")
    print(pd.DataFrame([human_mean]).T)
    return human_mean

# ====== 모델별 평균 비교 ======
def compare_means(final_df, human_mean, model_prefix):
    """
    모델별 평균값과 사람 평균값의 차이 계산
    - 사람 점수를 기준으로 모델의 상대적 차이 표시
    """
    print(f"\n=== {model_prefix} Metric별 평균 비교 ===")

    methods = [g for g in final_df["그룹"].unique() if g.startswith(model_prefix)]
    table_rows = []

    for metric in METRIC_KEYS:
        row = {"Metric": metric, "Human": round(human_mean.get(metric, 0.0), 4)}

        for m in methods:
            val = final_df.loc[final_df["그룹"] == m, metric].values
            if len(val) == 0:
                row[m] = "N/A"
                continue
            val = float(val[0])
            diff = val - human_mean.get(metric, 0.0)
            diff_symbol = "+" if diff >= 0 else ""
            row[m] = f"{val:.3f} ({diff_symbol}{diff:.3f})"

        table_rows.append(row)

    compare_df = pd.DataFrame(table_rows)
    compare_df = compare_df[["Metric", "Human"] + methods]
    print(compare_df)
    compare_df.to_csv(f"COMPARE_{model_prefix}_vs_Human.csv", index=False, encoding="utf-8-sig")
    print(f"=== {model_prefix} 비교 결과 저장 완료 ===")

    return compare_df

# ====== 실행 ======
if __name__ == "__main__":
    print("=== 사람 메트릭 vs 모델별 평균 비교 시작 ===")

    # 1. 사람 평균 메트릭 불러오기
    human_mean = load_human_metrics("Human_Metrics.csv")

    # 2. 모델 메트릭 불러오기
    final_df = pd.read_csv("/Users/khm/Desktop/Python/전략 있음/Negotiation_JSD.csv")

    # 3. 모델별 평균 비교 수행
    for model_prefix in ["Claude", "GPT4", "GPT35"]:
        compare_means(final_df, human_mean, model_prefix)



# import pandas as pd
# import numpy as np
# from scipy.stats import f_oneway
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import warnings

# warnings.filterwarnings("ignore")

# # ====== 설정 ======
# METRIC_KEYS = [
#     "AgreementRate",
#     "Fairness",
#     "Aggressiveness",
#     "Bias",
#     "ConcessionRate",
#     "RelativeEfficiency",
#     "ProbingRatio"
# ]

# # ====== 그룹 이름 매핑 ======
# GROUP_NAME_MAP = {
#     "Claude_Default": "Claude : Baseline",
#     "Claude_All": "Claude : CoT & IC",
#     "Claude_Medi_Default": "Claude : Mediator",
#     "Claude_Medi_All": "Claude : Mediator & CoT & IC",
#     "GPT35_Default": "GPT-3.5 : Baseline",
#     "GPT35_All": "GPT-3.5 : CoT & IC",
#     "GPT35_Medi_Default": "GPT-3.5 : Mediator",
#     "GPT35_Medi_All": "GPT-3.5 : Mediator & CoT & IC",
#     "GPT4_Default": "GPT-4 : Baseline",
#     "GPT4_All": "GPT-4 : CoT & IC",
#     "GPT4_Medi_Default": "GPT-4 : Mediator",
#     "GPT4_Medi_All": "GPT-4 : Mediator & CoT & IC",
# }


# # ====== 사람 데이터 불러오기 ======
# def load_human_data(csv_path):
#     df = pd.read_csv(csv_path)
#     df = df[df[METRIC_KEYS].notna().all(axis=1)]
#     df = df[df[METRIC_KEYS].sum(axis=1) > 0]
#     print(f"Loaded Human Data: {len(df)} rows")
#     return df


# # ====== LLM 메트릭 파일 불러오기 ======
# def load_llm_metrics(result_csv):
#     df = pd.read_csv(result_csv)
#     df = df[df[METRIC_KEYS].notna().all(axis=1)]
#     return df


# # ====== Metric별 ANOVA + Tukey HSD (Human과만 비교) ======
# def metric_anova_table(human_df, llm_df, model_prefix):
#     methods = [c for c in llm_df["그룹"].unique() if c.startswith(model_prefix)]
#     table_rows = []

#     for metric in METRIC_KEYS:
#         human_vals = human_df[metric].values
#         groups = [human_vals]
#         labels = ["Human"]

#         for m in methods:
#             vals = llm_df.loc[llm_df["그룹"] == m, metric].values
#             if len(vals) > 0:
#                 groups.append(vals)
#                 labels.append(m)

#         if len(groups) > 1:
#             try:
#                 f_val, p_val = f_oneway(*groups)
#             except Exception as e:
#                 print(f"Error in ANOVA for {metric}: {e}")
#                 f_val, p_val = np.nan, np.nan

#             # 전체 데이터 통합
#             all_scores, all_groups = [], []
#             for lbl, vals in zip(labels, groups):
#                 all_scores.extend(vals)
#                 all_groups.extend([lbl] * len(vals))

#             # Tukey로 Human과 유의하게 다른 그룹 찾기
#             sig_vs_human = set()
#             if len(set(all_groups)) > 1 and len(all_scores) > len(set(all_groups)):
#                 try:
#                     tukey = pairwise_tukeyhsd(endog=np.array(all_scores),
#                                               groups=np.array(all_groups),
#                                               alpha=0.05)
#                     tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
#                     for _, r in tukey_df.iterrows():
#                         g1, g2, reject = r["group1"], r["group2"], r["reject"]
#                         if "Human" in [g1, g2] and reject:
#                             sig_vs_human.add(g1 if g1 != "Human" else g2)
#                 except Exception as e:
#                     print(f"Tukey failed for {metric}: {e}")

#             # 결과 행 구성
#             row = {"Metric": metric, "F-value": round(f_val, 4), "p-value": round(p_val, 5)}
#             for lbl, vals in zip(labels, groups):
#                 mean_val = np.mean(vals)
#                 mapped_lbl = GROUP_NAME_MAP.get(lbl, lbl)  # 보기 좋게 이름 변경
#                 if lbl in sig_vs_human:
#                     mark = "**" if p_val < 0.01 else "*"
#                     row[mapped_lbl] = f"{mean_val:.3f}{mark}"
#                 else:
#                     row[mapped_lbl] = f"{mean_val:.3f}"

#             table_rows.append(row)

#     anova_df = pd.DataFrame(table_rows)
#     anova_df.to_csv(f"ANOVA_{model_prefix}_Metrics.csv", index=False, encoding="utf-8-sig")
#     print(f"\n=== {model_prefix} Metric별 ANOVA 완료 (CSV 저장) ===")
#     print(anova_df)
#     return anova_df


# # ====== 실행 ======
# if __name__ == "__main__":
#     human_csv = "/Users/khm/Desktop/Python/Human_Metrics.csv"
#     llm_result_csv = "/Users/khm/Desktop/Python/전략 있음/Negotiation_JSD.csv"
#     # llm_result_csv = "/Users/khm/Desktop/Python/전략 없음/Negotiation_JSD.csv"

#     human_df = load_human_data(human_csv)
#     llm_df = load_llm_metrics(llm_result_csv)

#     for model_prefix in ["Claude", "GPT35", "GPT4"]:
#         metric_anova_table(human_df, llm_df, model_prefix)

# import pandas as pd
# import numpy as np
# from scipy.stats import f_oneway, ttest_ind
# import warnings
# warnings.filterwarnings("ignore")

# # ====== 설정 ======
# METRIC_KEYS = [
#     "AgreementRate", "Fairness", "Aggressiveness", "Bias",
#     "ConcessionRate", "RelativeEfficiency", "ProbingRatio"
# ]
# MODEL_PREFIXES = ["Claude", "GPT35", "GPT4"]

# # ====== 모델별 평균 계산 ======
# def aggregate_model_means(df):
#     """각 모델별로 여러 메서드 평균 계산"""
#     agg_rows = []
#     for model in MODEL_PREFIXES:
#         model_df = df[df["그룹"].str.startswith(model)]
#         if len(model_df) == 0:
#             continue
#         row = {"Group": model}
#         for metric in METRIC_KEYS:
#             row[metric] = model_df[metric].mean()
#         agg_rows.append(row)
#     return pd.DataFrame(agg_rows)

# # ====== ANOVA + 사람과의 t-test ======
# def model_level_anova_ttest(human_df, llm_df):
#     """사람 vs 각 모델 평균 비교 (p-value에 유의성 표시 포함)"""
#     model_means = aggregate_model_means(llm_df)
#     human_vals = human_df[METRIC_KEYS]
#     human_mean = human_vals.mean()

#     table_rows = []

#     for metric in METRIC_KEYS:
#         # ANOVA (전체 평균 차이)
#         all_data = [human_vals[metric].values]
#         for model in MODEL_PREFIXES:
#             vals = llm_df.loc[llm_df["그룹"].str.startswith(model), metric].values
#             all_data.append(vals)
#         f_val, p_val_anova = f_oneway(*all_data)

#         # 행 초기화
#         row = {
#             "Metric": metric,
#             "F-value": round(f_val, 4),
#             "p-value(ANOVA)": round(p_val_anova, 5),
#             "Human_mean": round(human_mean[metric], 3)
#         }

#         # 각 모델별 t-test 수행
#         for model in MODEL_PREFIXES:
#             model_vals = llm_df.loc[llm_df["그룹"].str.startswith(model), metric].values
#             if len(model_vals) == 0:
#                 continue

#             t_stat, p_val = ttest_ind(human_vals[metric].values, model_vals, equal_var=False)
#             mean_val = np.mean(model_vals)
#             sig = "★★" if p_val < 0.01 else "★" if p_val < 0.05 else ""
#             pval_str = f"{p_val:.5f}{sig}"

#             row[f"{model}_mean"] = round(mean_val, 3)
#             row[f"{model}_pval"] = pval_str

#         table_rows.append(row)

#     result_df = pd.DataFrame(table_rows)
#     result_df.to_csv("ANOVA_ttest_mergedSig_No_Strat.csv", index=False, encoding="utf-8-sig")

#     print("\n=== Model-Level ANOVA + t-test (유의성 p-value 병합 버전) ===")
#     print(result_df)
#     return result_df


# # ====== 실행 ======
# if __name__ == "__main__":
#     human_csv = "/Users/khm/Desktop/Python/Human_Metrics.csv"
#     # llm_csv = "/Users/khm/Desktop/Python/전략 있음/Negotiation_JSD.csv"
#     llm_csv = "/Users/khm/Desktop/Python/전략 없음/Negotiation_JSD.csv"

#     human_df = pd.read_csv(human_csv)
#     llm_df = pd.read_csv(llm_csv)

#     model_level_anova_ttest(human_df, llm_df)

# import pandas as pd
# import numpy as np
# from scipy.stats import ttest_ind, f_oneway
# import warnings
# warnings.filterwarnings("ignore")

# # ====== 설정 ======
# METRIC_KEYS = [
#     "AgreementRate", "Fairness", "Aggressiveness", "Bias",
#     "ConcessionRate", "RelativeEfficiency", "ProbingRatio"
# ]
# MODEL_PREFIXES = ["Claude", "GPT35", "GPT4"]

# # ====== 전략 비교 (전략 있음 vs 없음) ======
# def strategy_comparison_ttest(llm_no_strat, llm_with_strat):
#     """전략 없음 vs 전략 있음 비교 (모델별로 t-test 수행, p-value 유의성 포함)"""
#     table_rows = []

#     for model in MODEL_PREFIXES:
#         model_no = llm_no_strat[llm_no_strat["그룹"].str.startswith(model)]
#         model_yes = llm_with_strat[llm_with_strat["그룹"].str.startswith(model)]

#         if len(model_no) == 0 or len(model_yes) == 0:
#             continue

#         for metric in METRIC_KEYS:
#             vals_no = model_no[metric].values
#             vals_yes = model_yes[metric].values

#             # ANOVA (참고용, 전체 차이 확인)
#             f_val, p_val_anova = f_oneway(vals_no, vals_yes)

#             # t-test (주요 비교)
#             t_stat, p_val = ttest_ind(vals_no, vals_yes, equal_var=False)
#             sig = "★★" if p_val < 0.01 else "★" if p_val < 0.05 else ""

#             mean_no = np.mean(vals_no)
#             mean_yes = np.mean(vals_yes)

#             row = {
#                 "Model": model,
#                 "Metric": metric,
#                 "No-Strategy_mean": round(mean_no, 3),
#                 "With-Strategy_mean": round(mean_yes, 3),
#                 "Diff": round(mean_yes - mean_no, 3),
#                 "F-value": round(f_val, 4),
#                 "p-value(ANOVA)": round(p_val_anova, 5),
#                 "p-value(t-test)": f"{p_val:.5f}{sig}"
#             }
#             table_rows.append(row)

#     result_df = pd.DataFrame(table_rows)
#     result_df.to_csv("Strategy_Comparison_ttest.csv", index=False, encoding="utf-8-sig")

#     print("\n=== Strategy-Level Comparison (전략 없음 vs 전략 있음) ===")
#     print(result_df)
#     return result_df


# # ====== 실행 ======
# if __name__ == "__main__":
#     base_dir = "/Users/khm/Desktop/Python"

#     llm_no_strat = pd.read_csv(f"{base_dir}/전략 없음/Negotiation_JSD.csv")
#     llm_with_strat = pd.read_csv(f"{base_dir}/전략 있음/Negotiation_JSD.csv")

#     strategy_comparison_ttest(llm_no_strat, llm_with_strat)
