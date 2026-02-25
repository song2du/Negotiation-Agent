## HCI Lab Negotiation Agent (Nego-interface)

This repository contains a Streamlit-based interface for **online negotiation experiments** in HCI Lab at Gachon University.
Participants take the role of buyer or seller and negotiate with an LLM-based agent.
The system logs the dialogue, outcomes, and subjective evaluations for research analysis.

---

### Main Features
- Streamlit web UI (setup screen + chat screen)
- LangGraph-based **negotiation workflow**
	- `setup → negotiator → logger`
- Multiple prompting strategies
	- `Baseline` – basic agent
	- `CoT_previous` – CoT + ICL
	- `CoT_upgrade` – CoT + ICL + few-shot + RAG + stronger negotiation strategy guidance
- RAG tool (`policy_search_tool`) for searching Naver Pay refund/exchange policies
- Firebase logging of negotiation traces and metadata
- Post-negotiation **human coding + subjective satisfaction questionnaire** UI

---

### Tech Stack
- Python 3.10+
- Streamlit
- LangChain / LangGraph
- OpenAI / Anthropic LLMs (e.g., gpt-4o, Claude 3 Sonnet)
- ChromaDB (vector DB for Naver Pay policy)
- Firebase Admin (for research data logging)

All required packages are listed in [requirements.txt](requirements.txt).

---

### How to Run (minimal)

This project is under active research development; details may change.
For now, a minimal way to start the app is:

```bash
pip install -r requirements.txt
streamlit run app.py
```

External services (Firebase, RAG DB, API keys, etc.) are configured
directly in the code (e.g., core/helpers.py, tools/rag_tools.py) and may evolve
without being fully reflected in this README.

---

### Logs & Research Data

- By default, negotiation results are stored in **Firebase Firestore**
	(collection `negotiation_results`), including:
	- full dialogue, per-turn metadata (speaker, utterance, agent "thought", tool calls)
	- buyer/seller goals, scores, rewards, human-coded outcome, survey responses
	- optional Pareto plot image as base64 (`pareto_image` field)
- Firebase requires a valid [serviceAccountKey.json](serviceAccountKey.json) and
	a reachable Firestore project; if initialization fails, results are **not** saved.
- A legacy CSV export helper exists but is not used by default in the current UI.

---

### Notes

- This codebase is a **research prototype**, not a production-ready system.
- For real user studies, please carefully review your IRB/ethics requirements,
	data retention policy, and any personal data handling.
