BC3415 IA — Retail Review Intelligence Demo
==========================================
https://www.youtube.com/watch?v=qgUEDD-lt94
==========================================


Turn thousands of raw e‑commerce reviews into trustworthy product insights using a lightweight Flask app, a fine‑tuned sentiment model, and an LLM assistant for explainability and auditability.

This repo contains:
- A minimal web UI to import reviews, initialize products, run automated checks, and review results
- A Flask backend with a SQLite database for quick local demos
- A fine‑tuned sequence classification model (DistilBERT) in the `model/` folder
- An optional GenAI step (Groq Llama‑3.3‑70B‑Versatile) to explain and validate suspect ratings
- A Colab notebook (`individual.ipynb`) showing the training and evaluation pipeline


Table of contents
-----------------
- Business case and approach
- System architecture
- Quick start (local)
- CSV format and data model
- Using the web app (3–5 min walkthrough cues)
- API endpoints
- Evaluation criteria mapping
- Extending to production
- Troubleshooting


Business case and approach
--------------------------
Problem: Retailers receive massive volumes of text reviews (often with images) daily. We need to convert this noisy signal into reliable insight for product, CX, and trust & safety teams.

Approach in this demo:
- Preprocess → classify sentiment with a fine‑tuned DistilBERT model (NEGATIVE/NEUTRAL/POSITIVE)
- Cross‑check the user’s star rating against the text sentiment to flag potential mismatches
- Optionally ask an LLM to explain “why this rating looks inappropriate” and provide a short reason
- Present results in a simple UI so a human can quickly review and confirm

Outcomes:
- Higher accuracy (DL model) + faster triage (UI) + better trust (LLM explanations)
- Modular, reusable components that you can swap or scale


System architecture
-------------------
- UI: `templates/index.html` (modern) and `templates/index_retro.html` (retro)
- Backend: `app.py` (Flask)
- Database: `database.db` (SQLite)
- Model: `model/` (HF format)
- LLM: Groq API (Llama‑3.3‑70B‑Versatile) for short reasoned judgments

Processing states per product (`products.check_flag`):
- 0 Not checked
- 1 Model checking
- 2 Ready for GenAI check
- 3 GenAI checking
- 4 Automatic checks done, pending review
- 5 All checks done


Quick start (local)
-------------------
Requirements
- Python 3.12+
- Linux/macOS/WSL (Windows works via WSL)

1) Create and activate a virtual environment
```bash
python3 -m venv bc3415_env
source .venv/bin/activate
```

2) Install dependencies & torch
```bash
pip install -r requirements.txt
```

3) (Optional) Configure Groq API key for GenAI step
```bash
export GROQ_API_KEY="<your_key>"
```
Note: The app can run without GenAI; the DL step works standalone.

4) Run the app
```bash
python app.py
```
Open http://127.0.0.1:5000/ (modern UI). Retro UI is at http://127.0.0.1:5000/retro.


CSV format and data model
-------------------------
Required CSV columns for `/loadCSV`:
```
reviewerID,asin,reviewerName,reviewText,overall,summary,unixReviewTime,reviewTime,helpful_yes,total_vote
```
Notes
- `overall` is the 1–5 star rating (int)
- `unixReviewTime` is an integer epoch time
- The app stores reviews in table `reviews` and products in `products`

SQLite schema (simplified)
```
products(asin PRIMARY KEY, image_link, title, seller, length, check_flag, num_checked, num_total)
reviews(uuid PRIMARY KEY, reviewerID, asin, reviewerName, reviewText, overall, summary,
		  unixReviewTime, reviewTime, helpful_yes, total_vote,
		  model_flag, genai_flag, genai_reason, user_flag)
```
Flags (per review)
- model_flag: 0 not processed, 1 pass, 2 fail (text contradicts rating)
- genai_flag: 0 not processed, 1 pass, 2 fail (LLM says rating is inappropriate)
- user_flag: 0 unreviewed, 1 showing (valid), -1 invalid/hidden


Using the web app (3–5 min walkthrough cues)
--------------------------------------------
Prep tips
- Trim the CSV to ~1,000 reviews for a single ASIN to keep the demo within 3–5 min
- Duplicate a product’s reviews and change the ASIN to simulate a new product
- Intentionally edit 1–2 ratings to be “wrong” for demonstration

Walkthrough flow (suggested)
1) Open the app at “modern UI” and state the problem briefly
	- Screen: Header and Import section
2) Import CSV (Upload → Import CSV)
	- Screen: Status chip shows progress; uninitialized products appear
3) Initialize product (click Initialize Product)
	- Screen: Fill Title/Image/Seller; save
4) Start Model Check
	- Screen: Product card shows progress bar and state badges
5) Start GenAI Check
	- Screen: After model completes, run GenAI; progress updates
6) Review flagged reviews
	- Screen: Open “All Reviews” or “Filtered”; show badges and reasons
7) (Optional) Human confirmation
	- Screen: Flag/Unflag; watch product state reach “All checks done”

Retro UI
- Visit `/retro` to quickly show the earlier UI for contrast.


API endpoints
-------------
GET
- `/` → modern UI
- `/retro` → retro UI
- `/allProducts` → list products and processing state
- `/uninitializedAsins` → ASINs in reviews without a product entry
- `/allReviews/<asin>` → all reviews for an ASIN
- `/modelFlag2Reviews/<asin>` → reviews with model_flag = 2
- `/genaiFlag2Reviews/<asin>` → reviews with genai_flag = 2

POST
- `/loadCSV` → multipart/form‑data with file field `file`
- `/initProduct` → JSON body `{ asin, image_link, title, seller, length }`

DELETE
- `/deleteProduct/<asin>` → deletes product and its reviews

Actions
- `/modelCheck/<asin>` → launch DL check in background
- `/genAiCheck/<asin>` → launch GenAI check in background
- `/userFlag/<uuid>/<asin>/<flag>` → set user_flag for a review (0, 1, or -1)


Evaluation criteria mapping
---------------------------
- Accuracy & performance
  - DistilBERT fine‑tuned classifier with truncation and dynamic padding
  - Async background processing to keep UI responsive
- Data preprocessing pipeline
  - See `individual.ipynb`: tokenization, cleaning, label mapping, train/test split
- Explainability & commentary
  - GenAI returns a brief reason for “inappropriate rating” decisions
  - UI surfaces status badges, counts, and reasons
- Creativity (text + image)
  - Demo focuses on text; image signals are planned as an extension (see next section)
- Usability & reusability
  - Minimal dependencies, simple CSV import, modular endpoints, and a clean UI


Extending to production
-----------------------
- Add image features: CLIP or ViT embeddings for image‑text consistency checks
- Confidence thresholds and calibration for model decisions
- Batch/stream processing with a job queue (Celery) and a Postgres backend
- Auth, audit logs, and role‑based approvals (human‑in‑the‑loop)
- BI integration: push aggregates to a dashboard (e.g., Metabase/Looker)


Troubleshooting
---------------
- CSV import fails: ensure all required columns exist and datatypes are valid
- “database is locked”: stop other processes or remove `database.db` to reset demo
- GenAI step fails: confirm `GROQ_API_KEY` is set and network access is available
- No products after import: check `/uninitializedAsins` or verify `asin` values


License
-------
CC-BY-SA @SyntaxaR
