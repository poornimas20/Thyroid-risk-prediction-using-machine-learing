"""
ThyroAI — app.py
3-class diagnosis: Hyperthyroid / Hypothyroid / Normal (real UCI diagnostic codes)
Risk level (No/Medium/High) derived from the model's own prediction confidence.
Features: age, sex, TSH, T3, T4 (real UCI data)
         + sleep_hours, stress_level, exercise_frequency, weight_gain (lifestyle)
"""

import os, json, sqlite3, io, datetime, warnings, functools, secrets
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)
app.jinja_env.filters['enumerate'] = enumerate
# SECRET_KEY signs the session cookie. Set THYROAI_SECRET_KEY in production;
# falling back to a random key means sessions reset every server restart.
app.secret_key = os.environ.get('THYROAI_SECRET_KEY', secrets.token_hex(32))

# ── Load artefacts ────────────────────────────────────────────────
MODEL  = joblib.load('model/best_model.pkl')
SCALER = joblib.load('model/scaler.pkl')
with open('model/feature_cols.json') as f:
    FEATURE_COLS = json.load(f)
with open('model/diagnosis_classes.json') as f:
    DIAG_CLASSES = json.load(f)   # e.g. ['Hyperthyroid', 'Hypothyroid', 'Normal']
with open('model/metrics.json') as f:
    METRICS = json.load(f)
with open('model/best_model_name.json') as f:
    _bmeta = json.load(f)
    BEST_MODEL_NAME = _bmeta['name']
    BEST_GB_PARAMS  = _bmeta.get('best_gb_params', {})
with open('model/label_mappings.json') as f:
    LABEL_MAPS = json.load(f)
with open('model/shap_available.json') as f:
    SHAP_AVAILABLE = json.load(f)['available']

# ── SQLite ────────────────────────────────────────────────────────
DB_PATH = 'thyroai.db'

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            patient_name TEXT,
            age INTEGER, sex TEXT,
            tsh REAL, t3 REAL, t4 REAL,
            sleep_hours INTEGER, stress_level TEXT,
            exercise_frequency TEXT, weight_gain INTEGER,
            diagnosis TEXT, confidence REAL, risk_level TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    con.commit(); con.close()
    # Lightweight migration for DBs created before this column existed
    con = sqlite3.connect(DB_PATH)
    existing_cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()]
    if 'patient_name' not in existing_cols:
        con.execute("ALTER TABLE predictions ADD COLUMN patient_name TEXT")
    if 'diagnosis' not in existing_cols:
        con.execute("ALTER TABLE predictions ADD COLUMN diagnosis TEXT")
    if 'confidence' not in existing_cols:
        con.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
    if 'risk_level' not in existing_cols:
        con.execute("ALTER TABLE predictions ADD COLUMN risk_level TEXT")
    if 't4' not in existing_cols:
        con.execute("ALTER TABLE predictions ADD COLUMN t4 REAL")
    con.commit(); con.close()

init_db()

# ── Auth helpers ──────────────────────────────────────────────────
def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.path))
        return view(*args, **kwargs)
    return wrapped

def get_user_by_username(username):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT id, username, password_hash FROM users WHERE username=?",
                       (username,)).fetchone()
    con.close()
    return row

def confidence_to_risk_level(diagnosis, confidence):
    """
    Risk level is derived purely from the model's own confidence in its
    predicted diagnosis (decision-driven, not a separate label):
      - If predicted diagnosis is Normal: risk is capped at Medium even at
        high confidence, since "Normal" itself is not a risk state.
      - If predicted diagnosis is Hypo/Hyperthyroid:
          confidence >= 0.80         → High Risk
          0.55 <= confidence < 0.80  → Medium Risk
          confidence < 0.55          → No Risk (model itself is unsure)
    """
    if diagnosis == 'Normal':
        return 'Medium Risk' if confidence < 0.60 else 'No Risk'
    if confidence >= 0.80:
        return 'High Risk'
    elif confidence >= 0.55:
        return 'Medium Risk'
    return 'No Risk'

def save_prediction(d):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO predictions
          (user_id,timestamp,patient_name,age,sex,tsh,t3,t4,sleep_hours,stress_level,
           exercise_frequency,weight_gain,diagnosis,confidence,risk_level)
        VALUES
          (:user_id,:timestamp,:patient_name,:age,:sex,:tsh,:t3,:t4,:sleep_hours,:stress_level,
           :exercise_frequency,:weight_gain,:diagnosis,:confidence,:risk_level)
    """, d)
    con.commit()
    pid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.close()
    return pid

# ── Auth Routes ───────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')
        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif get_user_by_username(username):
            flash('That username is already taken.', 'error')
        else:
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
                (username, generate_password_hash(password),
                 datetime.datetime.now().isoformat(timespec='seconds'))
            )
            con.commit(); con.close()
            flash('Account created — please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        row = get_user_by_username(username)
        if row and check_password_hash(row[2], password):
            session['user_id']  = row[0]
            session['username'] = row[1]
            nxt = request.args.get('next')
            return redirect(nxt or url_for('home'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    f = request.form
    try:
        # Collect raw form values
        patient_name = f.get('patient_name', '').strip() or 'Unnamed Patient'
        age          = int(f['age'])
        sex_str      = f['sex']                         # 'F' or 'M'
        tsh          = float(f['TSH'])
        t3           = float(f['T3'])
        t4           = float(f["T4"])
        sleep_hours  = int(f['sleep_hours'])
        stress_str   = f['stress_level']                # 'Low'/'Medium'/'High'
        exercise_str = f['exercise_frequency']          # 'Never'/'Weekly'/'Daily'
        weight_gain  = int(f['weight_gain'])
    except (KeyError, ValueError) as e:
        return f"Form error: {e}", 400

    # Encode exactly as training did
    sex_enc      = 1 if sex_str == 'F' else 0
    stress_enc   = LABEL_MAPS['stress_level'][stress_str]
    exercise_enc = LABEL_MAPS['exercise_frequency'][exercise_str]

    values_enc = {
        'age': age, 'sex': sex_enc, 'TSH': tsh, 'T3': t3, 'T4': t4,
        'sleep_hours': sleep_hours, 'stress_level': stress_enc,
        'exercise_frequency': exercise_enc, 'weight_gain': weight_gain
    }

    X = pd.DataFrame([values_enc])[FEATURE_COLS]
    X_s = SCALER.transform(X)
    proba = MODEL.predict_proba(X_s)[0]          # array aligned to DIAG_CLASSES order
    pred_idx   = int(np.argmax(proba))
    diagnosis  = DIAG_CLASSES[pred_idx]
    confidence = float(proba[pred_idx])
    risk_level = confidence_to_risk_level(diagnosis, confidence)

    # Per-class probabilities, for display (sorted high → low)
    class_probs = sorted(
        [(DIAG_CLASSES[i], round(float(proba[i]) * 100, 1)) for i in range(len(DIAG_CLASSES))],
        key=lambda x: x[1], reverse=True
    )

    pid = save_prediction({
        'user_id': session['user_id'],
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'patient_name': patient_name,
        'age': age, 'sex': sex_str,
        'tsh': tsh, 't3': t3, 't4': t4,
        'sleep_hours': sleep_hours,
        'stress_level': stress_str,
        'exercise_frequency': exercise_str,
        'weight_gain': weight_gain,
        'diagnosis': diagnosis,
        'confidence': round(confidence, 4),
        'risk_level': risk_level,
    })

    return render_template('result.html',
        patient_name=patient_name,
        diagnosis=diagnosis,
        risk_level=risk_level,
        class_probs=class_probs,
        values={'age':age,'sex':sex_str,'TSH':tsh,'T3':t3,'T4':t4,
                'sleep_hours':sleep_hours,'stress_level':stress_str,
                'exercise_frequency':exercise_str,'weight_gain':weight_gain},
        pred_id=pid,
        best_model=BEST_MODEL_NAME
    )

@app.route('/report/<int:pid>')
@login_required
def download_report(pid):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.units import cm

    con = sqlite3.connect(DB_PATH)
    cols = ['id','user_id','timestamp','patient_name','age','sex','tsh','t3','t4','sleep_hours',
            'stress_level','exercise_frequency','weight_gain','diagnosis','confidence','risk_level']
    row  = con.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                       (pid, session['user_id'])).fetchone()
    con.close()
    if not row: return "Not found", 404
    data = dict(zip(cols, row))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1  = ParagraphStyle('H1', parent=styles['Title'], fontSize=20, spaceAfter=6,
                          textColor=colors.HexColor('#0f172a'))
    h2  = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12,
                          spaceBefore=12, spaceAfter=4, textColor=colors.HexColor('#0f172a'))
    n   = styles['Normal']
    story = []

    story.append(Paragraph("ThyroAI — Thyroid Screening Report", h1))
    story.append(Paragraph(
        f"Patient: <b>{data['patient_name'] or 'Unnamed Patient'}</b>  |  "
        f"Generated: {data['timestamp']}  |  Record ID: #{data['id']}", n))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.4*cm))

    risk_color_map = {
        'High Risk':   colors.HexColor('#dc2626'),
        'Medium Risk': colors.HexColor('#d97706'),
        'No Risk':     colors.HexColor('#16a34a'),
    }
    rc = risk_color_map.get(data['risk_level'], colors.HexColor('#0f172a'))

    story.append(Paragraph("Predicted Diagnosis", h2))
    story.append(Paragraph(f'<font color="{rc.hexval()}" size="18"><b>{data["diagnosis"]}</b></font>', n))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Risk Level", h2))
    story.append(Paragraph(f'<font color="{rc.hexval()}" size="16"><b>{data["risk_level"]}</b></font>', n))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Thyroid Function Test (TFT) — Clinical & Lifestyle Parameters", h2))
    param_table = [
        ['Test Description','Value(s)','Reference Range'],
        ['Age',       str(data['age']),              '—'],
        ['Sex',       data['sex'],                   '—'],
        ['TSH (Thyroid Stimulating Hormone)', f"{data['tsh']} mIU/L",  '0.3 – 4.5'],
        ['T3 (Tri-iodothyronine) Total',      f"{data['t3']} nmol/L",  '1.1 – 2.5'],
        ['T4 (Thyroxine) Total',              f"{data['t4']} nmol/L",  '60 – 150'],
        ['Sleep',     f"{data['sleep_hours']} hrs/night", '7 – 9'],
        ['Stress',    data['stress_level'],           'Low'],
        ['Exercise',  data['exercise_frequency'],     'Daily/Weekly'],
        ['Weight Gain', 'Yes' if data['weight_gain'] else 'No', 'No'],
    ]
    t2 = Table(param_table, colWidths=[7*cm, 4.5*cm, 5.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#0d9488')),
        ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('GRID',      (0,0),(-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0fdfa'),colors.white]),
        ('FONTSIZE',  (0,0),(-1,-1), 9), ('LEFTPADDING',(0,0),(-1,-1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Recommendation", h2))
    if data['diagnosis'] == 'Hyperthyroid':
        rec = ("Indicators suggest possible hyperthyroidism (overactive thyroid). "
               "Consult an endocrinologist for a full TSH/T3/T4 blood panel and clinical evaluation.")
    elif data['diagnosis'] == 'Hypothyroid':
        rec = ("Indicators suggest possible hypothyroidism (underactive thyroid). "
               "Consult an endocrinologist for a full TSH/T3/T4 blood panel and clinical evaluation.")
    else:
        rec = ("Indicators are within the normal range. "
               "Maintain current habits and schedule routine annual check-ups.")
    story.append(Paragraph(rec, n))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "<i>ThyroAI is a screening tool only and does NOT constitute medical advice. "
        "Always consult a qualified healthcare professional.</i>", n))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf', as_attachment=True,
                     download_name=f'ThyroAI_Report_{pid}.pdf')

@app.route('/history')
@login_required
def history():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id,timestamp,patient_name,age,sex,tsh,diagnosis,risk_level "
        "FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 20",
        (session['user_id'],)
    ).fetchall()
    con.close()
    return render_template('history.html', rows=rows)

@app.route('/analytics')
@login_required
def analytics():
    con = sqlite3.connect(DB_PATH)
    uid = session['user_id']

    total = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id=?", (uid,)).fetchone()[0]
    high_risk = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id=? AND risk_level='High Risk'", (uid,)).fetchone()[0]
    medium_risk = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id=? AND risk_level='Medium Risk'", (uid,)).fetchone()[0]
    no_risk = total - high_risk - medium_risk

    avgs = con.execute(
        "SELECT AVG(tsh), AVG(sleep_hours) FROM predictions WHERE user_id=?", (uid,)
    ).fetchone()
    avg_tsh   = round(avgs[0], 2) if avgs[0] is not None else 0
    avg_sleep = round(avgs[1], 2) if avgs[1] is not None else 0

    # Last 20 predictions, oldest → newest, for the trend line.
    # We plot TSH (the primary clinical number) instead of an abstract model
    # confidence score — TSH is what the patient/professor can actually interpret,
    # and a shaded "normal range" band gives instant visual context.
    trend_rows = con.execute(
        "SELECT timestamp, tsh, risk_level FROM predictions WHERE user_id=? "
        "ORDER BY id DESC LIMIT 20", (uid,)
    ).fetchall()
    trend_rows = list(reversed(trend_rows))

    con.close()

    trend_labels = [r[0][5:16].replace('T', ' ') for r in trend_rows]  # MM-DD HH:MM
    trend_values = [round(r[1], 2) for r in trend_rows]                # raw TSH value
    trend_risk   = [r[2] for r in trend_rows]                          # risk level per point, for point color

    return render_template('analytics.html',
        total=total, high_risk=high_risk, medium_risk=medium_risk, no_risk=no_risk,
        avg_tsh=avg_tsh, avg_sleep=avg_sleep,
        trend_labels=trend_labels, trend_values=trend_values, trend_risk=trend_risk)

if __name__ == '__main__':
    app.run(debug=True)
