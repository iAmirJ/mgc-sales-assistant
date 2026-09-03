# MGC Sales Assistant & Lead Scoring

This project implements the four-part technical build task for MGC Developments. It tackles two major problems for the sales team:
1. **Answering repetitive questions accurately** via a Grounded Document AI.
2. **Prioritizing which leads to call** via a Machine Learning Lead Scoring model.

Below is a breakdown of how each part of the task was addressed, followed by installation instructions and working demo videos.

---

## Part 1: AI Development (Grounded Document Assistant)
A Streamlit-based chat interface allows salespeople to ask natural-language questions. Answers are **strictly grounded** in the three provided MGC documents (`docs/`).

### How it handles the 5 hard cases:
- **Base price of a 2-bed in Block B?** (Straight lookup): Retrieves the correct markdown chunk and cites the specific document and section.
- **Total for a Margalla-facing corner unit, floor 15, 2-bed Block B?** (Calculation): Extracts the base price and location premiums, then runs a deterministic calculation in Python to avoid LLM math hallucinations. It shows a transparent price breakdown in the UI.
- **What's the transfer fee?** (Conflict): The system detects that the documents disagree (2% vs 2.5%) and explicitly surfaces this conflict to the user rather than arbitrarily picking one.
- **Rental yield on a 1-bed?** (Abstention): The information isn't in the documents. The system safely refuses to answer ("I don't have that information") instead of inventing a number.
- **Who is the anchor tenant?** (Explicitly unconfirmed): The documents state no anchor tenant is confirmed; the assistant correctly relays this.

### Real-World Example

**Salesperson Question:**

> What is the base price of a 2-bed Standard unit in Block B?

**AI Assistant Answer:**

> The base price of a 2-bed Standard unit in Block B is PKR 22,425,000 for approximately 1,150 sq ft.
>
> **Source:** 02_price_list_payment_plan.md — Base Prices (Block B)

The answer is retrieved directly from the provided MGC documents rather than generated from external knowledge. This allows the salesperson to quickly verify the information and trace the answer back to its source.

---

## Part 2: Database (Schema & Queries)
The schema and queries for the messy CRM dump (`leads.csv`) are located in the `database/` folder.

- **Schema Decisions (`schema.sql`):** I used a single `leads` table. For a transactional CRM, flattening lead details (like source and city) into one table is highly performant and avoids unnecessary JOINs for simple lookups. I used robust typing (`DECIMAL` for currency/time, `BOOLEAN` flags).
- **Duplicate Prevention:** The CRM dump contains 320 duplicates. In a real database, this is prevented by adding a `UNIQUE` constraint on the `crm_record_hash`.
- **Queries (`queries.sql`):** 
  - **Query 1** calculates the conversion rate by source, specifically filtering for sources with `COUNT(*) >= 200` using a `HAVING` clause, and sorting by the best rate.
  - **Query 2** groups by `crm_record_hash` to identify and count duplicate leads.

---

## Part 3: ML Lead Scoring (Honest Baseline)
I built a model to predict the likelihood of a lead converting. The sales team can use this to prioritize high-value leads.

### Data Cleaning & Leakage Decisions
The most critical part of this task was preventing **Target Leakage**. 
- **What I kept:** Lead creation-time features (`source`, `city`, `area`, `property_type`, `budget`, `bedrooms`, `is_overseas`, etc.).
- **What I dropped (Leakage):** `token_amount_received_pkr`, `calls_made`, `total_call_seconds`, `whatsapp_replies`, `site_visits`. 
*Why?* If a lead has a "token amount received" or high "call seconds", they are already deep in the sales pipeline or have already converted. Training a model on this post-contact data creates a fake 99% accuracy model that is completely useless for scoring a *brand new* lead. I built an honest model that only uses day-zero information.

### Model & Metric
I trained a **GradientBoostingClassifier** (in `ml/train.py`). Because the dataset is heavily imbalanced (only ~6.9% of leads convert), traditional "Accuracy" is misleading (predicting "No" every time gives 93% accuracy). 
Instead, I optimized and reported on **PR-AUC (Average Precision)**, which directly measures the model's ability to identify true positives without being overwhelmed by the majority negative class.

### Business Scenario Example
When a new lead arrives from a "Billboard" in "Gujranwala" looking for a low-budget plot, the model (trained on historical patterns) might assign it a high probability (~71%). The sales team immediately sees **"High likelihood of conversion"** in the app and knows to call them first. Conversely, an over-budget lead from a poorly performing source will score low, allowing the team to deprioritize it.

### Real-World Example

**New Lead:**

A salesperson receives a new lead looking for a residential property. The lead provides information such as:

- Lead Source: Billboard
- City: Islamabad
- Property Type: Apartment
- Bedrooms: 2
- Budget: PKR 250 Lac
- Overseas: No
- Referred by Existing Client: Yes
- Financing Approved: Yes

The lead is passed to the trained ML pipeline using only information available when the lead is created.

**Example Model Result:**

> Conversion Probability: 71%

> Prediction: High likelihood of conversion

### What does 71% probability mean?

The model estimates that this lead has a relatively high likelihood of converting compared with leads that receive lower scores.

For the sales team, this means the lead can be prioritized for earlier follow-up.

The probability is **not a guarantee that the lead will convert**. It is a model-generated score based on patterns learned from historical CRM data.

For example:

- **Higher probability** → prioritize the lead for faster follow-up.
- **Medium probability** → follow up normally and monitor engagement.
- **Lower probability** → lower priority compared with higher-scoring leads.

The score should therefore be used as a **decision-support signal**, not as a guaranteed prediction of customer behavior.

---

## Part 4: Web Development
I tied everything together in a single **Streamlit** application (`app.py`). 
- **Main View:** The RAG assistant where the salesperson chats.
- **Sidebar:** The Lead Scoring form. A user can input lead details, and the ML model evaluates it synchronously in real-time without disrupting the chat state.

---

## Installation & Running Guide

### 1. Requirements
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Setup the AI Assistant
The RAG features require a Gemini API key.
1. Copy `.env.example` to `.env`
2. Add your Google AI Studio key: `GEMINI_API_KEY=your_key_here`

### 3. Run the ML Pipeline (Optional)
To retrain the ML model and view the evaluation metrics in your terminal:
```bash
python -m ml.train
```

### 4. Start the Application
Start the Streamlit server:
```bash
streamlit run app.py
```
Open `http://localhost:8501`. You can chat with the assistant in the main window and score leads using the sidebar.

---

## Working Demonstration Videos
Below are the screen recordings demonstrating the application in action:

- **RAG Assistant in action:** [recorded-sales-assistent.mp4](./recorded-sales-assistent.mp4)
- **ML Model & Integration:** [recorded-ML-model-accuracy-92.mp4](./recorded-ML-model-accuracy-92.mp4)

*(Note: Depending on your markdown viewer, you may need to download the .mp4 files from the repository to view them).*
