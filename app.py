import pandas as pd
from flask import Flask, g, jsonify, request
import sqlite3
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import uuid
from groq import Groq
import os



DATABASE = 'database.db'
os.environ['GROQ_API_KEY'] = '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'
MODEL_DIR = './model'
app = Flask(__name__)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
groq_client = Groq()

dataset2id = {1:0, 2:0, 3:1, 4:2, 5:2}
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2:"POSITIVE"}
label2id = {"NEGATIVE": 0, "NEUTRAL": 1,"POSITIVE": 2}

# DB Structure:
#
# CREATE TABLE "products" (
#   "asin" text NOT NULL,
#   "image_link" text,
#   "title" text,
#   "seller" text,
#   PRIMARY KEY ("asin")
# )
#
# CREATE TABLE "reviews" (
#   "uuid" text NOT NULL,
#   "reviewerID" text NOT NULL,
#   "asin" text NOT NULL,
#   "reviewerName" text NOT NULL,
#   "reviewText" text NOT NULL,
#   "overall" integer NOT NULL,
#   "summary" text NOT NULL,
#   "unixReviewTime" integer NOT NULL,
#   "reviewTime" text NOT NULL,
#   "helpful_yes" integer NOT NULL,
#   "total_vote" integer NOT NULL,
#   "model_flag" integer DEFAULT 0,
#   "genai_flag" integer DEFAULT 0,
#   "user_flag" integer DEFAULT 0,
#   PRIMARY KEY ("uuid")
# )

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # To access columns by name
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/modelCheck/<string:asin>') # 0: not checked, 1: normal, 2: potential wrongly-rated
def model_check(asin: str):
    db = get_db()
    cur = db.cursor()
    cur.execute('SELECT uuid, summary, reviewText, overall WHERE asin = ? AND model_flag = 0', (asin,))
    rows = cur.fetchall()
    # Use pipeline to validate
    for row in rows:
        combined_text = str(row[1]) + "\n" + str(row[2])
        result = classifier(combined_text)[0]
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        outputs = model(**inputs)
        predicted_class_id = outputs.logits.argmax().item()
        flag = 1 if dataset2id[row[3]] == predicted_class_id else 2
        cur.execute('UPDATE reviews SET model_flag = ? WHERE uuid = ?', (flag, row[0],))
        db.commit()
    return jsonify({'status': 'checked'}), 200
        

@app.route('/genAiCheck/<string:asin>') # 0: not checked, 1: normal, 2: potential wrongly-rated
def gen_ai_check(asin: str):
    db = get_db()
    cur = db.cursor()
    # Get product name from products table
    cur.execute('SELECT title FROM products WHERE asin = ?', (asin,))
    product_row = cur.fetchone()
    if not product_row:
        return jsonify({'error': 'ASIN not found in products'}), 404
    product_name = product_row[0]
    cur.execute('SELECT uuid, summary, reviewText, overall FROM reviews WHERE asin = ? AND model_flag = 2 AND genai_flag = 0', (asin,))
    rows = cur.fetchall()
    return jsonify({'status': 'checked'}), 200

async def groq_check (name: str, rows: list):
    db = get_db()
    cur = db.cursor()
    for row in rows:
        r = groq_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at identifying if user as rated wrongly from their review on a online marketspace."},
                {"role": "user", "content": f"Given the product name `{name}`, the user gives the review summary `{row[1]}` and fulltext `{row[2]}`, the user gives a product score {row[3]} out of 5. Determine if the the user have made a mistake when rating. Respond with only the number 1 if the rating seems correct, or 2 if the rating seems incorrect. DO NOT RESPOND WITH ANYTHING ELSE, ONLY 1 OR 2."}
            ],
        )
        flag = int(r.choices[0].message.content.strip())
        if flag not in [1, 2]:
            print("Unexpected response from Groq:", r.choices[0].message.content)
            flag = 2
        cur.execute('UPDATE reviews SET genai_flag = ? WHERE uuid = ?', (flag, row[0],))
        db.commit()

@app.route('/userFlag/<string:uuid>/<int:flag>') # 0: unflag, -1: invalid, 1: valid
def user_flag(uuid: str, flag: int):
    db = get_db()
    cur = db.cursor()
    cur.execute('UPDATE reviews SET user_flag = ? WHERE uuid = ?', (flag, uuid,))
    db.commit()
    return jsonify({'status': 'flagged'}), 200

@app.route('/clearAll/<asin: str>') # Clear all flags for a given ASIN
def clear_all(asin: str):
    db = get_db()
    cur = db.cursor()
    cur.execute('UPDATE reviews SET model_flag = 0, genai_flag = 0, user_flag = 0 WHERE asin = ?', (asin,))
    db.commit()
    return jsonify({'status': 'cleared'}), 200

@app.route('/allReviews/<string:asin>')
def all_reviews(asin: str):
    db = get_db()
    cur = db.cursor()
    cur.execute('SELECT uuid, reviewerID, asin, reviewerName, reviewText, overall, summary, unixReviewTime, reviewTime, helpful_yes, total_vote, model_flag, genai_flag, user_flag FROM reviews WHERE asin = ?', (asin,))
    rows = cur.fetchall()
    reviews = [dict(row) for row in rows]
    return jsonify(reviews)

@app.route('/loadCSV', methods=['POST'])
def load_csv():
    # Expect a multipart/form-data upload with field name 'file'
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'no selected file'}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'failed to read CSV: {e}'}), 400

    # Required columns per schema
    required = [
        'reviewerID', 'asin', 'reviewerName', 'reviewText',
        'overall', 'summary', 'unixReviewTime', 'reviewTime',
        'helpful_yes', 'total_vote'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({'error': 'missing required columns', 'missing': missing}), 400

    db = get_db()
    cur = db.cursor()
    inserted = 0
    try:
        for _, row in df.iterrows():
            uid = str(uuid.uuid4())
            # Safely coerce numeric fields where appropriate
            try:
                overall = int(row['overall']) if not pd.isna(row['overall']) else None
            except:
                print("Error converting overall:", row['overall'])
                overall = None
            try:
                unix_time = int(row['unixReviewTime']) if not pd.isna(row['unixReviewTime']) else None
            except:
                print("Error converting unix time:", row['unixReviewTime'])
                unix_time = None
            try:
                helpful_yes = int(row['helpful_yes']) if not pd.isna(row['helpful_yes']) else 0
            except:
                print("Error converting helpful_yes:", row['helpful_yes'])
                helpful_yes = 0
            try:
                total_vote = int(row['total_vote']) if not pd.isna(row['total_vote']) else 0
            except:
                print("Error converting total_vote:", row['total_vote'])
                total_vote = 0

            params = (
                uid,
                row['reviewerID'],
                row['asin'],
                row['reviewerName'],
                row['reviewText'],
                overall,
                row['summary'],
                unix_time,
                row['reviewTime'],
                helpful_yes,
                total_vote,
                0,  # model_flag
                0,  # genai_flag
                0   # user_flag
            )
            cur.execute(
                '''INSERT INTO reviews
                   (uuid, reviewerID, asin, reviewerName, reviewText, overall, summary, unixReviewTime, reviewTime, helpful_yes, total_vote, model_flag, genai_flag, user_flag)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                params
            )
            inserted += 1
        db.commit()
    except Exception as e:
        db.rollback()
        return jsonify({'error': f'database error: {e}'}), 500

    return jsonify({'inserted': inserted}), 200

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)