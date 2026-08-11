import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Harmonia | Music & Mental Health",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional design system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');

:root {
    --ink: #1a2332;
    --ink-soft: #3d4a5c;
    --muted: #6b7a8f;
    --line: #d8e0ea;
    --surface: #ffffff;
    --surface-soft: #f4f7fb;
    --teal: #0d9488;
    --teal-deep: #0f766e;
    --teal-soft: #ccfbf1;
    --coral: #e11d48;
    --coral-soft: #ffe4e6;
    --amber: #d97706;
    --amber-soft: #fef3c7;
    --sky: #0284c7;
    --radius: 14px;
    --shadow: 0 8px 28px rgba(26, 35, 50, 0.08);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, #d9f5f1 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #e8eef8 0%, transparent 50%),
        linear-gradient(180deg, #f7fafc 0%, #eef3f8 100%);
    color: var(--ink);
}

.main .block-container {
    max-width: 1120px;
    padding: 1.5rem 1.75rem 4rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1c2e 0%, #152a40 55%, #12353a 100%) !important;
    border-right: none !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.25rem 1rem 2rem;
}

[data-testid="stSidebar"] * {
    color: #e8eef6 !important;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: #9fb0c6 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #fff !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #fff !important;
}

.brand-mark {
    font-family: 'Source Serif 4', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.25rem;
}

.brand-sub {
    color: #8fa3bc !important;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    line-height: 1.45;
}

.dev-credit {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.12);
    font-size: 0.78rem;
    color: #8fa3bc !important;
    letter-spacing: 0.02em;
}

.dev-credit strong {
    color: #5eead4 !important;
    font-weight: 600;
}

/* Hero */
.app-hero {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    margin-bottom: 1.75rem;
    padding: 1.75rem 0 1.5rem;
    border-bottom: 1px solid var(--line);
    animation: riseIn 0.55s ease both;
}

.app-hero .eyebrow {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal);
}

.app-hero h1 {
    font-family: 'Source Serif 4', serif;
    font-size: clamp(1.9rem, 3.2vw, 2.6rem);
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0;
}

.app-hero p {
    color: var(--ink-soft);
    font-size: 1.05rem;
    max-width: 42rem;
    margin: 0;
    line-height: 1.55;
}

.section-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.45rem;
    font-weight: 650;
    color: var(--ink);
    margin: 1.75rem 0 0.85rem;
    letter-spacing: -0.02em;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 1.25rem 1.35rem;
    box-shadow: var(--shadow);
    margin: 0.85rem 0 1.15rem;
    animation: riseIn 0.5s ease both;
}

.panel-teal { border-left: 4px solid var(--teal); }
.panel-amber { border-left: 4px solid var(--amber); background: linear-gradient(90deg, #fffbeb 0%, #fff 40%); }
.panel-rose { border-left: 4px solid var(--coral); background: linear-gradient(90deg, #fff1f2 0%, #fff 40%); }
.panel-sky { border-left: 4px solid var(--sky); }

.panel h3 {
    font-family: 'Source Serif 4', serif;
    font-size: 1.15rem;
    margin: 0 0 0.45rem;
    color: var(--ink);
}

.panel p, .panel li {
    color: var(--ink-soft);
    line-height: 1.55;
    margin: 0.35rem 0;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--teal-deep) !important;
    font-weight: 750 !important;
}

/* Forms & controls */
.stForm {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 1.35rem 1.4rem 1.1rem;
    box-shadow: var(--shadow);
}

.form-block {
    background: var(--surface-soft);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.85rem;
}

.form-block h4 {
    font-family: 'Source Serif 4', serif;
    color: var(--ink) !important;
    font-size: 1.05rem;
    margin: 0 0 0.75rem;
}

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--ink) !important;
    font-weight: 550 !important;
}

.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
    background: #fff !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-deep) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important;
    font-weight: 650 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 20px rgba(13, 148, 136, 0.28) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 24px rgba(13, 148, 136, 0.35) !important;
}

/* Result badges */
.result-chip {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 650;
    margin-top: 0.35rem;
}

.chip-high { background: var(--coral-soft); color: var(--coral); }
.chip-low { background: var(--teal-soft); color: var(--teal-deep); }

.footer-bar {
    margin-top: 2.5rem;
    padding: 1.1rem 0 0.25rem;
    border-top: 1px solid var(--line);
    text-align: center;
    color: var(--muted);
    font-size: 0.88rem;
}

.footer-bar strong {
    color: var(--teal-deep);
    font-weight: 700;
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .app-hero h1 { font-size: 1.7rem; }
    .main .block-container { padding: 1rem 1rem 3rem; }
}
</style>
""", unsafe_allow_html=True)


def inject_footer():
    st.markdown(
        """
        <div class="footer-bar">
            Harmonia — Music &amp; Mental Health Predictor<br/>
            Developed by <strong>MUHAMMAD OBAID</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load data and create models
@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv('preprocessed_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'preprocessed_dataset.csv' is in the same directory.")
        return None


@st.cache_data
def create_models(df):
    """Create and train advanced machine learning models using techniques from third.py"""
    if df is None:
        return None, None, None, None, None, None, None, None, None, None

    # Advanced feature engineering (same as third.py)
    print("Creating advanced features...")

    # 1. Music Engagement Features
    df['Music_Intensity_Score'] = (df['Hours per day'] * df['Total_Frequency_Score']).fillna(0)
    df['Genre_Diversity_Score'] = df[[col for col in df.columns if 'Frequency' in col]].std(axis=1).fillna(0)
    df['High_Engagement_Genres'] = (df[['Frequency [Rock]', 'Frequency [Metal]', 'Frequency [EDM]']].sum(axis=1))
    df['Calm_Genres'] = (df[['Frequency [Classical]', 'Frequency [Jazz]', 'Frequency [Lofi]']].sum(axis=1))

    # 2. Behavioral Pattern Features
    df['Work_Music_Interaction'] = df['While working'] * df['Hours per day']
    df['Exploratory_Engagement'] = df['Exploratory'] * df['Music_Engagement_Score']
    df['Instrumental_Engagement'] = df['Instrumentalist'] * df['Total_Frequency_Score']

    # 3. Age-based Features
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 25, 30, 100], labels=['Teen', 'Young_Adult', 'Adult', 'Senior'])
    df['Age_Music_Interaction'] = df['Age'] * df['Hours per day']

    # 4. Music Therapy Indicators
    df['Music_Therapy_Score'] = (df['Music effects'] * df['Music_Engagement_Score']).fillna(0)
    df['Positive_Music_Effect'] = (df['Music effects'] > 0).astype(int)

    # 5. Genre Emotional Profiles
    df['Aggressive_Genres'] = (df['Frequency [Metal]'] + df['Frequency [Rock]'] + df['Frequency [Rap]']).fillna(0)
    df['Emotional_Genres'] = (df['Frequency [Pop]'] + df['Frequency [R&B]'] + df['Frequency [Folk]']).fillna(0)
    df['Energetic_Genres'] = (df['Frequency [EDM]'] + df['Frequency [Hip hop]'] + df['Frequency [K pop]']).fillna(0)

    # 6. Music Listening Patterns
    df['Evening_Listener'] = (df['Hours per day'] > df['Hours per day'].median()).astype(int)
    df['Heavy_Listener'] = (df['Hours per day'] > df['Hours per day'].quantile(0.75)).astype(int)
    df['Diverse_Listener'] = (df['Genre_Diversity_Score'] > df['Genre_Diversity_Score'].median()).astype(int)

    # Create comprehensive feature set
    base_features = [
        'Hours per day', 'BPM', 'Total_Frequency_Score', 'Music_Engagement_Score',
        'While working', 'Exploratory', 'Instrumentalist', 'Composer', 'Foreign languages',
        'Age'
    ]

    # Add genre features
    genre_columns = [col for col in df.columns if 'Frequency' in col]

    # Add engineered features
    engineered_features = [
        'Music_Intensity_Score', 'Genre_Diversity_Score', 'High_Engagement_Genres', 'Calm_Genres',
        'Work_Music_Interaction', 'Exploratory_Engagement', 'Instrumental_Engagement',
        'Age_Music_Interaction', 'Music_Therapy_Score', 'Positive_Music_Effect',
        'Aggressive_Genres', 'Emotional_Genres', 'Energetic_Genres',
        'Evening_Listener', 'Heavy_Listener', 'Diverse_Listener'
    ]

    # Add age group dummy variables
    age_dummies = pd.get_dummies(df['Age_Group'], prefix='Age_Group')
    df = pd.concat([df, age_dummies], axis=1)

    # Combine all features
    all_features = base_features + genre_columns + engineered_features + age_dummies.columns.tolist()
    feature_columns = [col for col in all_features if col in df.columns]

    # Prepare data
    X = df[feature_columns].fillna(0)
    y = df['High_Mental_Health_Risk']

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Advanced models with regularization (same as third.py)
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import VotingClassifier

    # Train advanced models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10,
                                              min_samples_leaf=5, max_features='sqrt', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                      max_depth=3, subsample=0.8, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_split=10,
                                           min_samples_leaf=5, max_features='sqrt', random_state=42),
        'SVM': SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, alpha=0.01,
                                       random_state=42)
    }

    # Ensemble model
    ensemble_models = [
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_split=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=50, max_depth=4, min_samples_split=10, random_state=42))
    ]

    ensemble = VotingClassifier(ensemble_models, voting='soft')
    models['Ensemble'] = ensemble

    # Train all models
    results = {}
    for name, model in models.items():
        if name in ['SVM', 'Neural Network']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': accuracy}

    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']

    # Cross-validation
    from sklearn.model_selection import cross_val_score
    if best_model_name in ['SVM', 'Neural Network']:
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

    return (best_model, results['Random Forest']['model'], results['Gradient Boosting']['model'],
            results['Ensemble']['model'], scaler, feature_columns,
            results['Random Forest']['accuracy'], results['Gradient Boosting']['accuracy'],
            results['Ensemble']['accuracy'], cv_scores.mean())


def predict_mental_health_risk(user_input, model, scaler, feature_columns):
    """Make prediction using the trained model"""
    # Create DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Ensure all required columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[feature_columns]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    return prediction, probability


PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Outfit, sans-serif', color='#1a2332'),
    margin=dict(l=20, r=20, t=48, b=20),
    colorway=['#0d9488', '#0284c7', '#d97706', '#e11d48', '#6366f1'],
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def main():
    # Sidebar brand + navigation
    with st.sidebar:
        st.markdown('<div class="brand-mark">Harmonia</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-sub">Music habits meet mental wellness — research-backed insights.</div>',
            unsafe_allow_html=True,
        )
        page = st.selectbox(
            "Navigate",
            ["Home", "Predictor", "Data Insights", "Research", "Feedback", "About"],
            label_visibility="visible",
        )
        st.markdown(
            """
            <div class="dev-credit">
                Developed by<br/><strong>MUHAMMAD OBAID</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load data
    df = load_data()
    if df is None:
        return

    # Create models
    with st.spinner("Preparing models…"):
        best_model, rf_model, gb_model, ensemble_model, scaler, feature_columns, rf_acc, gb_acc, ensemble_acc, cv_acc = create_models(df)

    if rf_model is None:
        st.error("Failed to create models. Please check your data.")
        return

    if page == "Home":
        show_home_page(df)
    elif page == "Predictor":
        show_predictor_page(best_model, rf_model, gb_model, ensemble_model, scaler, feature_columns, rf_acc, gb_acc, ensemble_acc, cv_acc)
    elif page == "Data Insights":
        show_data_insights(df)
    elif page == "Research":
        show_research_findings()
    elif page == "Feedback":
        show_feedback_page()
    elif page == "About":
        show_about_page()

    inject_footer()


def show_home_page(df):
    """Display the home page with overview"""
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Music &amp; Mental Health</div>
            <h1>Understand how your listening habits relate to wellbeing</h1>
            <p>An AI research tool trained on 736 participants — explore patterns, run a personal prediction, and review evidence-based findings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Participants", f"{len(df):,}", "Research sample")
    with col2:
        high_risk = df['High_Mental_Health_Risk'].sum()
        st.metric("High-risk cases", f"{high_risk}", f"{high_risk/len(df)*100:.1f}% of sample")
    with col3:
        st.metric("Avg. anxiety", f"{df['Anxiety'].mean():.1f}", "Scale 0–10")
    with col4:
        st.metric("Avg. depression", f"{df['Depression'].mean():.1f}", "Scale 0–10")

    st.markdown(
        """
        <div class="panel panel-teal">
            <h3>What this tool does</h3>
            <p>Harmonia analyzes music listening habits to surface patterns linked with mental health indicators.
            Insights are correlational and educational — not a clinical diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Research highlights</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="panel">
                <h3>Music &amp; mental health</h3>
                <ul>
                    <li>Daily listening hours correlate with depression</li>
                    <li>Rock &amp; Metal associate with higher depression scores</li>
                    <li>Classical listening shows protective signals</li>
                    <li>Engagement score is a strong predictor</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="panel">
                <h3>Model performance</h3>
                <ul>
                    <li>61.1% mental-health risk accuracy</li>
                    <li>75.7% genre-based prediction accuracy</li>
                    <li>70.3% exploratory-behavior accuracy</li>
                    <li>Regularized models to avoid overfitting</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel panel-sky">
            <h3>Next step</h3>
            <p>Open <strong>Predictor</strong> in the sidebar to enter your listening profile and receive a personalized risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_predictor_page(best_model, rf_model, gb_model, ensemble_model, scaler, feature_columns, rf_acc, gb_acc, ensemble_acc, cv_acc):
    """Display the mental health predictor page"""
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Assessment</div>
            <h1>Mental health risk predictor</h1>
            <p>Share your listening profile. Multiple models evaluate patterns and return a consensus assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Random Forest", f"{rf_acc*100:.1f}%", "Accuracy")
    with col2:
        st.metric("Gradient Boosting", f"{gb_acc*100:.1f}%", "Accuracy")
    with col3:
        st.metric("Ensemble", f"{ensemble_acc*100:.1f}%", "Accuracy")
    with col4:
        st.metric("Cross-validation", f"{cv_acc*100:.1f}%", "Robust accuracy")

    st.markdown(
        """
        <div class="panel panel-amber">
            <h3>Important disclaimer</h3>
            <p><strong>For educational and research use only.</strong> This is not a substitute for professional mental health care.
            If you are struggling, please speak with a qualified clinician.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Your music habits</div>', unsafe_allow_html=True)

    with st.form("music_habits_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="form-block"><h4>Basics</h4>', unsafe_allow_html=True)
            age = st.slider("Age", 10, 80, 25, help="Your current age")
            hours_per_day = st.slider("Hours of music per day", 0.0, 12.0, 2.0, 0.5, help="Average daily listening hours")
            bpm = st.slider("Preferred BPM", 60, 200, 120, help="Typical tempo you prefer")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="form-block"><h4>Engagement</h4>', unsafe_allow_html=True)
            while_working = st.selectbox("Listen while working?", ["No", "Yes"])
            exploratory = st.selectbox("Explore new music often?", ["No", "Yes"])
            instrumentalist = st.selectbox("Are you an instrumentalist?", ["No", "Yes"])
            composer = st.selectbox("Do you compose music?", ["No", "Yes"])
            foreign_languages = st.selectbox("Listen to foreign-language music?", ["No", "Yes"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(
                '<div class="form-block"><h4>Genre frequency</h4><p style="color:#6b7a8f;font-size:0.9rem;margin-top:0;">0 = Never · 3 = Very often</p>',
                unsafe_allow_html=True,
            )
            genre_frequencies = {}
            genres = [
                "Rock", "Pop", "Hip hop", "Rap", "Metal", "Classical", "Jazz", "Folk",
                "R&B", "EDM", "K pop", "Lofi", "Video game music"
            ]
            for genre in genres:
                genre_frequencies[f"Frequency [{genre}]"] = st.slider(
                    genre, 0, 3, 1, key=f"freq_{genre}",
                    help=f"How often do you listen to {genre}?"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        total_frequency_score = sum(genre_frequencies.values())
        music_engagement_score = (hours_per_day + total_frequency_score / 10 + bpm / 100) / 3
        submitted = st.form_submit_button("Generate prediction", type="primary", use_container_width=True)

    if submitted:
        user_input = {
            'Age': age,
            'Hours per day': hours_per_day,
            'BPM': bpm,
            'Total_Frequency_Score': total_frequency_score,
            'Music_Engagement_Score': music_engagement_score,
            'While working': 1 if while_working == "Yes" else 0,
            'Exploratory': 1 if exploratory == "Yes" else 0,
            'Instrumentalist': 1 if instrumentalist == "Yes" else 0,
            'Composer': 1 if composer == "Yes" else 0,
            'Foreign languages': 1 if foreign_languages == "Yes" else 0
        }
        user_input.update(genre_frequencies)

        st.markdown('<div class="section-title">Model results</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            rf_pred, rf_prob = predict_mental_health_risk(user_input, rf_model, scaler, feature_columns)
            st.metric("Random Forest", f"{rf_acc*100:.1f}%", "Model accuracy")
            risk_level = "High Risk" if rf_pred == 1 else "Low Risk"
            confidence = rf_prob[1] if rf_pred == 1 else rf_prob[0]
            chip = "chip-high" if rf_pred == 1 else "chip-low"
            st.markdown(f'<span class="result-chip {chip}">{risk_level} · {confidence*100:.1f}%</span>', unsafe_allow_html=True)

        with col2:
            gb_pred, gb_prob = predict_mental_health_risk(user_input, gb_model, scaler, feature_columns)
            st.metric("Gradient Boosting", f"{gb_acc*100:.1f}%", "Model accuracy")
            risk_level = "High Risk" if gb_pred == 1 else "Low Risk"
            confidence = gb_prob[1] if gb_pred == 1 else gb_prob[0]
            chip = "chip-high" if gb_pred == 1 else "chip-low"
            st.markdown(f'<span class="result-chip {chip}">{risk_level} · {confidence*100:.1f}%</span>', unsafe_allow_html=True)

        with col3:
            ensemble_pred, ensemble_prob = predict_mental_health_risk(user_input, ensemble_model, scaler, feature_columns)
            st.metric("Ensemble", f"{ensemble_acc*100:.1f}%", "Model accuracy")
            risk_level = "High Risk" if ensemble_pred == 1 else "Low Risk"
            confidence = ensemble_prob[1] if ensemble_pred == 1 else ensemble_prob[0]
            chip = "chip-high" if ensemble_pred == 1 else "chip-low"
            st.markdown(f'<span class="result-chip {chip}">{risk_level} · {confidence*100:.1f}%</span>', unsafe_allow_html=True)

        best_pred, best_prob = predict_mental_health_risk(user_input, best_model, scaler, feature_columns)
        predictions = [rf_pred, gb_pred, ensemble_pred]
        avg_prediction = np.mean(predictions)
        overall_risk = "High Risk" if avg_prediction > 0.5 else "Low Risk"

        st.markdown('<div class="section-title">Overall assessment</div>', unsafe_allow_html=True)
        if overall_risk == "High Risk":
            st.markdown(
                """
                <div class="panel panel-rose">
                    <h3>Higher mental health risk signal</h3>
                    <p>Based on listening patterns, the models suggest elevated risk indicators. Possible contributors:</p>
                    <ul>
                        <li>Extended daily listening hours</li>
                        <li>Preference for Rock, Metal, or Rap</li>
                        <li>Specific engagement and therapy-effect patterns</li>
                    </ul>
                    <p><strong>Recommendation:</strong> Consider speaking with a mental health professional for personalized support.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="panel panel-teal">
                    <h3>Lower mental health risk signal</h3>
                    <p>Your listening profile aligns with lower risk indicators. Common factors include:</p>
                    <ul>
                        <li>Balanced daily consumption</li>
                        <li>Diverse genre exposure</li>
                        <li>Healthy engagement patterns</li>
                    </ul>
                    <p><strong>Keep going</strong> — music can remain a positive part of your wellness routine.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Personalized notes</div>', unsafe_allow_html=True)
        insights = []
        if hours_per_day > 4:
            insights.append("High daily listening may associate with mental health challenges")
        if genre_frequencies.get("Frequency [Rock]", 0) > 2:
            insights.append("Strong Rock preference is linked with higher depression risk in this dataset")
        if genre_frequencies.get("Frequency [Classical]", 0) > 2:
            insights.append("Classical preference may show protective effects")
        if total_frequency_score > 20:
            insights.append("High overall genre frequency suggests exploratory listening")
        if while_working == "Yes":
            insights.append("Listening while working is common and not strongly linked to risk here")

        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.markdown("- Your habits look balanced — no strong risk flags from the simple rules.")


def show_data_insights(df):
    """Display data insights and visualizations"""
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Analytics</div>
            <h1>Data insights &amp; visualizations</h1>
            <p>Explore distributions, listening patterns, and correlations from the research sample.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Anxiety distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x='Anxiety', nbins=20, title='Anxiety levels')
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with col2:
            st.markdown('<div class="section-title">Depression distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x='Depression', nbins=20, title='Depression levels')
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown('<div class="section-title">Listening patterns</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, y='Hours per day', title='Hours of music per day')
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with col2:
            genre_cols = [col for col in df.columns if 'Frequency' in col]
            genre_means = df[genre_cols].mean().sort_values(ascending=False)
            fig = px.bar(
                x=genre_means.values,
                y=genre_means.index,
                orientation='h',
                title='Average genre preferences'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown('<div class="section-title">Music–mental health correlations</div>', unsafe_allow_html=True)
        correlation_vars = [
            'Hours per day', 'BPM', 'Total_Frequency_Score', 'Music_Engagement_Score',
            'Anxiety', 'Depression', 'Mental_Health_Severity'
        ]
        corr_matrix = df[correlation_vars].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation matrix",
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown('<div class="section-title">Mental health by working status</div>', unsafe_allow_html=True)
        working_analysis = df.groupby('While working')[['Anxiety', 'Depression']].mean()
        fig = px.bar(
            working_analysis,
            title='Average scores by working status',
            labels={'index': 'Working status', 'value': 'Average score'}
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading visualizations: {str(e)}")
        st.info("Please ensure the dataset columns are available for charting.")


def show_research_findings():
    """Display research findings from the assignments"""
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Evidence</div>
            <h1>Research findings</h1>
            <p>Answers to the core research questions, with model performance and study limits.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel panel-teal">
            <h3>Methodology</h3>
            <p>Analysis of 736 participants linking music habits to mental health using statistical tests and regularized machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Key questions</div>', unsafe_allow_html=True)

    findings = [
        (
            "Does hours per day correlate with mental health?",
            "Yes — hours per day correlates with depression (r = 0.130, p < 0.001).",
            [
                "Higher listening hours associate with higher depression",
                "Correlation is stronger for depression than anxiety",
                "Total frequency score is a key predictor",
            ],
        ),
        (
            "Which genres relate to mental health?",
            "Rock, Metal, and Rap show positive correlations with depression.",
            [
                "Rock & Metal: strongest links (r = 0.193, r = 0.177)",
                "Classical: protective signals for anxiety",
                "Genre-based prediction accuracy: 75.7%",
            ],
        ),
        (
            "Does BPM affect mental health?",
            "BPM alone is weak; genre energy matters more.",
            [
                "BPM is not a strong standalone predictor",
                "High-energy genres associate more with anxiety",
                "Energy profile outweighs tempo",
            ],
        ),
        (
            "Do working listeners differ?",
            "Working status has minimal impact on outcomes.",
            [
                "No significant anxiety/depression gap",
                "Working listeners tend to be more exploratory",
                "Both groups favor Rock and Pop",
            ],
        ),
        (
            "Are exploratory listeners more stable?",
            "Exploratory behavior shows a marginal depression association.",
            [
                "No clear anxiety difference",
                "Higher music diversity among explorers",
                "Engagement is the strongest exploratory predictor",
            ],
        ),
    ]

    for title, answer, bullets in findings:
        items = "".join(f"<li>{b}</li>" for b in bullets)
        st.markdown(
            f"""
            <div class="panel">
                <h3>{title}</h3>
                <p><strong>{answer}</strong></p>
                <ul>{items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mental health risk", "61.1%", "Cross-validation")
    with col2:
        st.metric("Genre prediction", "75.7%", "High accuracy")
    with col3:
        st.metric("Exploratory behavior", "70.3%", "Good accuracy")

    st.markdown(
        """
        <div class="panel panel-sky">
            <h3>Techniques used</h3>
            <ul>
                <li>17+ engineered features without leakage</li>
                <li>Random Forest, Gradient Boosting, Ensemble</li>
                <li>Regularization for realistic accuracy</li>
                <li>5-fold cross-validation</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - BPM data was standardized, limiting tempo analysis  
        - Self-reported responses may include bias  
        - Limited demographic spread  
        - No temporal listening sequences  
        - Moderate risk-model accuracy (61.1%)
        """
    )


def show_feedback_page():
    """Display feedback collection page"""
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Improve</div>
            <h1>Share your feedback</h1>
            <p>Tell us what worked, what didn’t, and what you’d like next.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel panel-teal">
            <h3>Help refine Harmonia</h3>
            <p>Your notes guide usability, clarity, and research presentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_type = st.selectbox(
                "What best describes you?",
                ["Student", "Professional", "Researcher", "General User", "Other"]
            )
            age_group = st.selectbox(
                "Age group",
                ["Under 18", "18-25", "26-35", "36-45", "46-55", "56-65", "Over 65"]
            )
        with col2:
            experience_level = st.selectbox(
                "Technical experience",
                ["Beginner", "Intermediate", "Advanced", "Expert"]
            )
            usage_frequency = st.selectbox(
                "How often do you use tools like this?",
                ["First time", "Rarely", "Sometimes", "Frequently", "Daily"]
            )

        st.markdown('<div class="section-title">Your experience</div>', unsafe_allow_html=True)
        usability_rating = st.slider("Ease of use (1–5)", 1, 5, 3)
        accuracy_rating = st.slider("Prediction usefulness (1–5)", 1, 5, 3)
        relevance_rating = st.slider("Insight relevance (1–5)", 1, 5, 3)

        st.markdown('<div class="section-title">Comments</div>', unsafe_allow_html=True)
        what_liked = st.text_area("What did you like most?", height=90)
        what_improved = st.text_area("What could be improved?", height=90)
        suggestions = st.text_area("Feature ideas?", height=90)
        general_feedback = st.text_area("Anything else?", height=90)

        submitted = st.form_submit_button("Submit feedback", type="primary", use_container_width=True)

    if submitted:
        feedback_data = {
            'timestamp': pd.Timestamp.now(),
            'user_type': user_type,
            'age_group': age_group,
            'experience_level': experience_level,
            'usage_frequency': usage_frequency,
            'usability_rating': usability_rating,
            'accuracy_rating': accuracy_rating,
            'relevance_rating': relevance_rating,
            'what_liked': what_liked,
            'what_improved': what_improved,
            'suggestions': suggestions,
            'general_feedback': general_feedback
        }

        try:
            feedback_df = pd.read_csv('user_feedback.csv')
            new_feedback = pd.DataFrame([feedback_data])
            feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        except FileNotFoundError:
            feedback_df = pd.DataFrame([feedback_data])

        feedback_df.to_csv('user_feedback.csv', index=False)
        st.success("Thank you — your feedback was saved.")

        st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
        st.markdown(f"**Usability:** {usability_rating}/5")
        st.markdown(f"**Usefulness:** {accuracy_rating}/5")
        st.markdown(f"**Relevance:** {relevance_rating}/5")
        if what_liked:
            st.markdown(f"**Liked:** {what_liked}")
        if what_improved:
            st.markdown(f"**Improve:** {what_improved}")


def show_about_page():
    """Display about page with project information"""
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Project</div>
            <h1>About Harmonia</h1>
            <p>A research application exploring links between music listening and mental health indicators.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel panel-teal">
            <h3>Purpose</h3>
            <p>Harmonia uses data science and machine learning to surface patterns that may support awareness
            and music-informed wellbeing conversations — never clinical diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Source:** Kaggle Music & Mental Health Survey Results  
        - **Participants:** 736  
        - **Features:** 33+ variables (demographics, habits, mental health)  
        - **Quality:** Fully preprocessed  
        - **Age range:** 10–89 (median 21)
        """
    )

    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Assignment 1 — Preprocessing:** imputation, encoding, feature engineering, outliers  

        **Assignment 2 — EDA:** descriptive stats, correlations, visualization, study design  

        **Assignment 3 — Modeling:** hypothesis tests, ML models, feature importance, evaluation
        """
    )

    st.markdown('<div class="section-title">Stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **UI:** Streamlit  
        - **Data:** Pandas, NumPy  
        - **ML:** Scikit-learn  
        - **Charts:** Plotly, Matplotlib, Seaborn  
        - **Stats:** SciPy  

        **Models:** Random Forest, Gradient Boosting, Logistic Regression, Ensemble methods
        """
    )

    st.markdown('<div class="section-title">Key findings</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Hours per day correlates with depression (r = 0.130, p < 0.001)  
        2. Rock and Metal associate with higher depression  
        3. Classical shows protective signals for anxiety  
        4. Music engagement is a strong risk predictor  
        5. Exploratory listening has a marginal depression link
        """
    )

    st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - Mental health risk: **61.1%** (regularized, realistic)  
        - Genre-based: **75.7%**  
        - Exploratory behavior: **70.3%**  
        - Evaluation: 5-fold cross-validation
        """
    )

    st.markdown('<div class="section-title">Limits & next steps</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Limits:** self-report bias, limited demographics, no temporal streams, moderate risk accuracy  

        **Next:** streaming telemetry, longitudinal tracking, deeper models, clinical validation
        """
    )

    st.markdown(
        """
        <div class="panel">
            <h3>Credit</h3>
            <p>This application was <strong>developed by MUHAMMAD OBAID</strong>.
            For questions or suggestions, use the Feedback section. Educational use only — not a clinical tool.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
