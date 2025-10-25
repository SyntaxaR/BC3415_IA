import pandas as pd
from flask import Flask, g, jsonify, request, render_template
import sqlite3
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import uuid
from groq import Groq
import os
import threading
import time

DATABASE = 'database.db'
os.environ['GROQ_API_KEY'] = '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'
MODEL_DIR = 'model'
app = Flask(__name__)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
groq_client = Groq()

dataset2id = {1:0, 2:0, 3:1, 4:2, 5:2}
dataset2label = {1:"NEGATIVE", 2:"NEGATIVE", 3:"NEUTRAL", 4:"POSITIVE", 5:"POSITIVE"}
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2:"POSITIVE"}
label2id = {"NEGATIVE": 0, "NEUTRAL": 1,"POSITIVE": 2}

@app.route('/')
def index():
    return render_template('index.html')

# DB Structure:
#
# CREATE TABLE "products" (
#   "asin" text NOT NULL,
#   "image_link" text,
#   "title" text,
#   "seller" text,
#   "length" integer DEFAULT 0,
#   "check_flag" integer DEFAULT 0, 0: unchecked, 1: model checking, 2: model check done, 3: genai checking, 4: genai check done, 5: all done
#   "num_checked" integer DEFAULT 0,
#   "num_total" integer DEFAULT 0,
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

def new_db_connection():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
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
    cur.execute('SELECT uuid, summary, reviewText, overall FROM reviews WHERE asin = ? AND model_flag = 0', (asin,))
    rows = cur.fetchall()
    if not rows:
        return jsonify({'status': 'no unprocessed reviews'}), 200
    # Flag
    # Use pipeline to validate
    threading.Thread(target=pretrained_model_check, args=(asin, rows)).start()
    return jsonify({'status': 'Model is checking...'}), 200
        
def pretrained_model_check(asin:str, rows: list):
    db = new_db_connection()
    cur = db.cursor()
    # Check if the check_flag is 0
    cur.execute('SELECT check_flag FROM products WHERE asin = ?', (asin,))
    product_row = cur.fetchone()
    if not product_row or product_row[0] != 0:
        return
    # Set check_flag to 1 and num_checked to 0 (model checking)
    cur.execute('UPDATE products SET check_flag = 1, num_checked = 0, num_total = ? WHERE asin = ?', (len(rows), asin,))
    db.commit()
    for index, row in enumerate(rows):
        # Update num_checked for every 20 reviews
        if index % 20 == 19:
            cur.execute('UPDATE products SET num_checked = ? WHERE asin = ?', (index + 1, asin,))
            db.commit()
        combined_text = str(row[1]) + "\n" + str(row[2])
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        flag = 1 if dataset2id[row[3]] == predicted_class_id else 2
        cur.execute('UPDATE reviews SET model_flag = ? WHERE uuid = ?', (flag, row[0],))
        db.commit()
    # Set check_flag to 2 (model check done)
    cur.execute('UPDATE products SET check_flag = 2 WHERE asin = ?', (asin,))
    db.commit()
    db.close()
    
@app.route('/userCompleteCheck/<string:asin>') # User indicates that model checking is complete
def user_complete_check(asin: str):
    db = get_db()
    cur = db.cursor()
    cur.execute('UPDATE products SET check_flag = 5 WHERE asin = ?', (asin,))
    db.commit()
    return jsonify({'status': 'success'}), 200

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
    threading.Thread(target=groq_check, args=(asin, product_name, rows)).start()
    return jsonify({'status': 'groq checking...'}), 200

def groq_check(asin: str, name: str, rows: list):
    db = new_db_connection()
    cur = db.cursor()
    # Check if the check_flag is 2
    cur.execute('SELECT check_flag FROM products WHERE asin = ?', (asin,))
    product_row = cur.fetchone()
    if not product_row or product_row[0] != 2:
        return
    # Set check_flag to 3 and num_checked to 0 (model checking)
    cur.execute('UPDATE products SET check_flag = 3, num_checked = 0, num_total = ? WHERE asin = ?', (len(rows), asin,))
    db.commit()
    for index, row in enumerate(rows):
        cur.execute('UPDATE products SET num_checked = ? WHERE asin = ?', (index + 1, asin,))
        db.commit()
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": """
                 You are evaluating whether a product rating is significantly unfair to the seller.
                 Determine if the rating is significantly unfair by checking:
                 1. **Obvious Mismatches Only**: Is there a clearly positive review (praises functionality, value, reliability) with a 1-2 star rating? Is there a clearly negative review (product failed, doesn't work, major defects) with a 4-5 star rating?
                 2. **Illegitimate Complaints**: Does the review penalize the product for inherent characteristics of the entire product category (e.g., "SD card is too small", "phone needs charging", "wireless earbuds need pairing")? Does the review blame the product for user error or misunderstanding?
                 3. **Tolerance for Rating**: If you think the rating is slightly off but not egregiously so (e.g., a 4-star review with minor complaints rated as 3 stars), consider it acceptable.
                 4. **Rules for Neutral Reviews (rating 3/5)**: Rating 3/5 is acceptable for mixed reviews, but only if seller or the product does have a problem according to rule 2, even if complaints seem minor; Rating 3/5 is unacceptable if user is complaining mainly on hypothetical concerns which has not happened.
                 You will be rewarded if the final output matches the agent's judgment.
                 Respond with only the number 1 if the rating is acceptable, or 2 if the rating is unacceptable. DO NOT RESPOND WITH ANYTHING ELSE, ONLY 1 OR 2.
                 """},
                {"role": "user", "content": f"Product name: `{name}`\n User review summary: `{row[1]}`\nUser's full review: `{row[2]}`\nUser's rating: {row[3]}/5\n OUTPUT:"}
            ],
        )
        flag = int(r.choices[0].message.content.strip())
        if flag not in [1, 2]:
            print("Unexpected response from Groq:", r.choices[0].message.content)
            flag = 2
        cur.execute('UPDATE reviews SET genai_flag = ? WHERE uuid = ?', (flag, row[0],))
        db.commit()
        time.sleep(1) # To avoid free plan rate limit
    # Set check_flag to 4 (genai check done)
    cur.execute('UPDATE products SET check_flag = 4 WHERE asin = ?', (asin,))
    db.commit()

@app.route('/userFlag/<string:uuid>/<int:flag>') # 0: unflag, -1: invalid, 1: valid
def user_flag(uuid: str, flag: int):
    db = get_db()
    cur = db.cursor()
    cur.execute('UPDATE reviews SET user_flag = ? WHERE uuid = ?', (flag, uuid,))
    db.commit()
    return jsonify({'status': 'flagged'}), 200

@app.route('/clearAllFlags/<string:asin>') # Clear all flags for a given ASIN
def clear_all_flags(asin: str):
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

@app.route('/modelFlag2Reviews/<string:asin>')
def model_flag2_reviews(asin: str):
    db = get_db()
    cur = db.cursor()
    cur.execute('SELECT uuid, reviewerID, asin, reviewerName, reviewText, overall, summary, unixReviewTime, reviewTime, helpful_yes, total_vote, model_flag, genai_flag, user_flag FROM reviews WHERE asin = ? AND model_flag = 2', (asin,))
    rows = cur.fetchall()
    reviews = [dict(row) for row in rows]
    return jsonify(reviews)

@app.route('/genaiFlag2Reviews/<string:asin>')
def genai_flag2_reviews(asin: str):
    db = get_db()
    cur = db.cursor()
    cur.execute('SELECT uuid, reviewerID, asin, reviewerName, reviewText, overall, summary, unixReviewTime, reviewTime, helpful_yes, total_vote, model_flag, genai_flag, user_flag FROM reviews WHERE asin = ? AND genai_flag = 2', (asin,))
    rows = cur.fetchall()
    reviews = [dict(row) for row in rows]
    return jsonify(reviews)

# Return ASINs in reviews table that are not in products table and number of reviews for each ASIN
@app.route('/uninitializedAsins')
def uninitialized_asins():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT asin, COUNT(*) FROM reviews WHERE asin NOT IN (SELECT asin FROM products) GROUP BY asin")
    rows = cur.fetchall()
    result = [{'asin': row[0], 'review_count': row[1]} for row in rows]
    return jsonify(result)

@app.route('/allProducts')
def all_products():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT asin, image_link, title, seller, length, check_flag, num_checked, num_total FROM products")
    rows = cur.fetchall()
    products = [{} for row in rows]
    for i, row in enumerate(rows):
        products[i] = {
            'asin': row[0],
            'image_link': row[1],
            'title': row[2],
            'seller': row[3],
            'length': row[4],
            'check_flag': row[5],
            'num_checked': row[6],
            'num_total': row[7]
        }
    return jsonify(products)

# fetch('/initProduct', {
#         method: 'POST',
#         headers: {'Content-Type': 'application/json'},
#         body: JSON.stringify({asin, image_link, title, seller, length})
#     }).then(res => res.json()).then(data => {
#         if (data.status === 'ok') {
#             bootstrap.Modal.getInstance(document.getElementById('productModal')).hide();
#             fetchProducts();
#         } else {
#             alert('Failed to initialize product: ' + (data.error || 'Unknown error'));
#         }
#     });

@app.route('/initProduct', methods=['POST'])
def init_product():
    data = request.json
    asin = data.get('asin')
    image_link = data.get('image_link')
    title = data.get('title')
    seller = data.get('seller')
    length = data.get('length')

    db = get_db()
    cur = db.cursor()
    cur.execute('INSERT INTO products (asin, image_link, title, seller, length) VALUES (?, ?, ?, ?, ?)',
                (asin, image_link, title, seller, length))
    db.commit()
    return jsonify({'status': 'ok'}), 200

@app.route('/deleteProduct/<string:asin>', methods=['DELETE'])
def delete_product(asin: str):
    db = get_db()
    cur = db.cursor()
    cur.execute('DELETE FROM reviews WHERE asin = ?', (asin,))
    cur.execute('DELETE FROM products WHERE asin = ?', (asin,))
    db.commit()
    return jsonify({'status': 'deleted'}), 200

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