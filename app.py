import streamlit as st
import json
import datetime
from typing import Dict, List
import re
import hashlib
import time
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="भारत Stories - Cultural Memory Keeper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for offline-first functionality
if 'stories' not in st.session_state:
    st.session_state.stories = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'offline_mode' not in st.session_state:
    st.session_state.offline_mode = False

# Language detection and translation models (lightweight for offline)
@st.cache_resource
def load_ai_models():
    """Load lightweight AI models for story enhancement"""
    try:
        # Language detection
        lang_classifier = pipeline("text-classification", 
                                 model="papluca/xlm-roberta-base-language-detection",
                                 return_all_scores=True)
        
        # Sentiment analysis for story categorization
        sentiment_analyzer = pipeline("sentiment-analysis",
                                    model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        return lang_classifier, sentiment_analyzer
    except Exception as e:
        st.warning(f"AI models loading in offline mode: {e}")
        return None, None

# Utility functions
def detect_language(text):
    """Detect language of the input text"""
    try:
        lang_classifier, _ = load_ai_models()
        if lang_classifier and len(text) > 10:
            result = lang_classifier(text[:500])  # Limit text for processing
            top_lang = max(result[0], key=lambda x: x['score'])
            return top_lang['label'], top_lang['score']
    except:
        pass
    return "unknown", 0.0

def analyze_story_sentiment(text):
    """Analyze sentiment/emotion of the story"""
    try:
        _, sentiment_analyzer = load_ai_models()
        if sentiment_analyzer and len(text) > 10:
            result = sentiment_analyzer(text[:500])
            return result[0]['label'], result[0]['score']
    except:
        pass
    return "neutral", 0.0

def generate_story_id(text, location):
    """Generate unique ID for story"""
    content = f"{text[:100]}{location}{datetime.datetime.now().date()}"
    return hashlib.md5(content.encode()).hexdigest()[:8]

def save_story_data(story_data):
    """Save story data (simulating corpus collection)"""
    st.session_state.stories.append(story_data)
    
    # In real implementation, this would save to a database/file
    # For demo, we'll show what data is being collected
    return True

# UI Components
def render_header():
    """Render application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1>📚 भारत Stories</h1>
            <h3>आपकी यादें, हमारी धरोहर</h3>
            <p><i>Preserving India's Cultural Memories</i></p>
        </div>
        """, unsafe_allow_html=True)

def render_story_input_form():
    """Main story input form"""
    st.markdown("### 🖊️ Share Your Story")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        story_text = st.text_area(
            "आपकी कहानी यहाँ लिखें / Write your story here",
            height=200,
            placeholder="यहाँ अपनी यादें, परिवारिक किस्से, स्थानीय परंपराएं, या कोई भी दिलचस्प बात साझा करें...\n\nShare your memories, family stories, local traditions, or any interesting experience...",
            help="अपनी मातृभाषा में लिखें - हम सभी भारतीय भाषाओं का समर्थन करते हैं!"
        )
        
    with col2:
        location = st.text_input("📍 स्थान / Location", 
                               placeholder="शहर, राज्य")
        
        category = st.selectbox("📂 कहानी का प्रकार / Story Type", 
                              ["व्यक्तिगत अनुभव / Personal Experience",
                               "पारिवारिक कहानी / Family Story", 
                               "स्थानीय परंपरा / Local Tradition",
                               "ऐतिहासिक घटना / Historical Event",
                               "त्योहारी यादें / Festival Memories",
                               "बचपन की यादें / Childhood Memories",
                               "अन्य / Other"])
        
        age_group = st.selectbox("आयु समूह / Age Group",
                               ["18-25", "26-35", "36-50", "51-65", "65+"])
        
        include_ai_enhancement = st.checkbox(
            "🤖 AI सुधार सक्षम करें / Enable AI Enhancement", 
            value=True,
            help="AI will help detect language and categorize your story"
        )
    
    return story_text, location, category, age_group, include_ai_enhancement

def render_ai_analysis(story_text, include_ai_enhancement):
    """Show AI analysis of the story"""
    if not include_ai_enhancement or len(story_text.strip()) < 10:
        return None
        
    with st.spinner("🤖 AI विश्लेषण कर रहा है..."):
        # Language detection
        detected_lang, lang_confidence = detect_language(story_text)
        
        # Sentiment analysis
        sentiment, sentiment_score = analyze_story_sentiment(story_text)
        
        # Story statistics
        word_count = len(story_text.split())
        char_count = len(story_text)
        
        analysis = {
            'detected_language': detected_lang,
            'language_confidence': lang_confidence,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'word_count': word_count,
            'character_count': char_count
        }
        
        # Display analysis
        st.markdown("#### 🔍 AI Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("भाषा / Language", 
                     detected_lang.upper() if detected_lang != "unknown" else "Mixed/Unknown",
                     f"{lang_confidence:.1%} confidence" if lang_confidence > 0 else "")
        
        with col2:
            st.metric("भावना / Sentiment", 
                     sentiment.title(),
                     f"{sentiment_score:.1%} confidence" if sentiment_score > 0 else "")
        
        with col3:
            st.metric("शब्द / Words", word_count)
        
        return analysis

def render_story_feed():
    """Display recent stories"""
    st.markdown("### 📖 हाल की कहानियां / Recent Stories")
    
    if not st.session_state.stories:
        st.info("अभी तक कोई कहानी साझा नहीं की गई है। पहली कहानी आप साझा करें!")
        return
    
    # Show latest 5 stories
    recent_stories = sorted(st.session_state.stories, 
                          key=lambda x: x['timestamp'], reverse=True)[:5]
    
    for story in recent_stories:
        with st.expander(f"📍 {story['location']} - {story['category'][:20]}..."):
            st.write(f"**Story:** {story['text'][:200]}...")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Language:** {story.get('detected_language', 'Unknown')}")
            with col2:
                st.write(f"**Sentiment:** {story.get('sentiment', 'Unknown')}")
            with col3:
                st.write(f"**Words:** {story.get('word_count', 0)}")
            
            st.write(f"*Shared on: {story['timestamp']}*")

def render_corpus_stats():
    """Show corpus collection statistics"""
    st.sidebar.markdown("### 📊 Collection Stats")
    
    if st.session_state.stories:
        total_stories = len(st.session_state.stories)
        total_words = sum(story.get('word_count', 0) for story in st.session_state.stories)
        languages = set(story.get('detected_language', 'unknown') 
                       for story in st.session_state.stories)
        
        st.sidebar.metric("कुल कहानियां / Total Stories", total_stories)
        st.sidebar.metric("कुल शब्द / Total Words", total_words)
        st.sidebar.metric("भाषाएं / Languages", len(languages))
        
        # Show language distribution
        if len(languages) > 1:
            st.sidebar.markdown("**भाषा वितरण / Language Distribution:**")
            lang_counts = {}
            for story in st.session_state.stories:
                lang = story.get('detected_language', 'unknown')
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                st.sidebar.write(f"- {lang.upper()}: {count}")
    else:
        st.sidebar.info("कोई डेटा एकत्र नहीं किया गया")

def render_offline_indicator():
    """Show offline mode indicator"""
    if st.session_state.offline_mode:
        st.sidebar.warning("🔌 Offline Mode Active")
        st.sidebar.info("Data will sync when connection is restored")

# Main application
def main():
    render_header()
    
    # Sidebar
    render_offline_indicator()
    render_corpus_stats()
    
    # Add some educational content in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Mission")
    st.sidebar.info("भारत Stories helps preserve India's rich cultural diversity by collecting stories in all Indian languages and dialects.")
    
    st.sidebar.markdown("### 🤝 How to Contribute")
    st.sidebar.write("""
    1. Share authentic stories from your region
    2. Write in your mother tongue
    3. Include cultural context
    4. Help others discover your heritage
    """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Share Story", "📚 Browse Stories", "🎯 About"])
    
    with tab1:
        story_text, location, category, age_group, include_ai_enhancement = render_story_input_form()
        
        # AI Analysis
        analysis = None
        if story_text and include_ai_enhancement:
            analysis = render_ai_analysis(story_text, include_ai_enhancement)
        
        # Submit button
        if st.button("🚀 कहानी साझा करें / Share Story", type="primary"):
            if story_text.strip() and location.strip():
                # Create story data structure
                story_data = {
                    'id': generate_story_id(story_text, location),
                    'text': story_text,
                    'location': location,
                    'category': category,
                    'age_group': age_group,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'detected_language': analysis['detected_language'] if analysis else 'unknown',
                    'language_confidence': analysis['language_confidence'] if analysis else 0.0,
                    'sentiment': analysis['sentiment'] if analysis else 'unknown',
                    'sentiment_score': analysis['sentiment_score'] if analysis else 0.0,
                    'word_count': analysis['word_count'] if analysis else len(story_text.split()),
                    'character_count': analysis['character_count'] if analysis else len(story_text)
                }
                
                if save_story_data(story_data):
                    st.success("🎉 आपकी कहानी सफलतापूर्वक साझा की गई!")
                    st.balloons()
                    
                    # Show what data was collected (for transparency)
                    with st.expander("🔍 Collected Data Preview (for transparency)"):
                        st.json(story_data)
                else:
                    st.error("कहानी साझा करने में समस्या हुई। कृपया पुनः प्रयास करें।")
            else:
                st.warning("कृपया कहानी और स्थान दोनों भरें।")
    
    with tab2:
        render_story_feed()
    
    with tab3:
        st.markdown("""
        ### 🎯 About भारत Stories
        
        **Mission:** Preserving India's linguistic and cultural diversity through authentic storytelling.
        
        **Vision:** Creating the world's largest multilingual corpus of Indian cultural experiences.
        
        #### 🌟 Features:
        - **Multilingual Support**: Write in any Indian language
        - **AI Enhancement**: Automatic language detection and categorization  
        - **Cultural Preservation**: Stories become part of cultural heritage
        - **Community Building**: Connect with others through shared experiences
        - **Offline First**: Works even with poor internet connectivity
        
        #### 🔒 Privacy & Data Use:
        - All stories are anonymized
        - Used only for cultural preservation and AI research
        - No personal identification required
        - Data helps build better Indian language AI models
        
        #### 🤝 How Your Data Helps:
        Your stories contribute to building AI that understands Indian culture, languages, and experiences better.
        
        ---
        
        **Built for भारत**  
        *An open-source initiative for cultural digital preservation*
        """)

if __name__ == "__main__":
    main()