"""
Smart Travel Planner - Beautiful UI with ChromaDB RAG
Compact code + Modern design + Full functionality
"""
import streamlit as st
import requests
from datetime import datetime
import re
import PyPDF2
from PIL import Image
import pytesseract
import chromadb
import hashlib
import os

# Page config
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    div.block-container {
        padding-top: 2rem;
    }
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .attraction-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 18px;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        color: #333;
    }
    .step-indicator {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="big-font">✈️ AI Travel Planner</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🤖 Powered by ChromaDB RAG • Smart Ticket Parser • Budget Planning</p>', unsafe_allow_html=True)

# ==================== CHROMADB SETUP ====================

@st.cache_resource
def init_db():
    """Initialize ChromaDB"""
    os.makedirs("./chroma_db", exist_ok=True)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    collections = {}
    for name in ["tickets", "attractions", "reviews"]:
        try:
            collections[name] = client.get_or_create_collection(name)
        except:
            collections[name] = client.create_collection(name)
    
    return collections

db = init_db()

# ==================== HELPER FUNCTIONS ====================

def parse_ticket(file):
    """Extract text from PDF/Image and parse travel info"""
    if file.type == "application/pdf":
        text = "".join([page.extract_text() for page in PyPDF2.PdfReader(file).pages])
    else:
        text = pytesseract.image_to_string(Image.open(file))
    
    info = {"destination": None, "date": None, "from": None, "to": None}
    dates = re.findall(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', text)
    if dates:
        info["date"] = dates[0]
    
    codes = re.findall(r'\b([A-Z]{3})\b', text)
    if len(codes) >= 2:
        info["from"], info["to"] = codes[0], codes[1]
        info["destination"] = codes[1]
    
    to_match = re.search(r'to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text, re.I)
    if to_match:
        info["destination"] = to_match.group(1)
    
    return info

def calc_days(date1, date2):
    """Calculate days between two dates"""
    formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d %b %Y']
    d1 = d2 = None
    
    for fmt in formats:
        try:
            d1 = datetime.strptime(date1, fmt)
            break
        except:
            continue
    
    for fmt in formats:
        try:
            d2 = datetime.strptime(date2, fmt)
            break
        except:
            continue
    
    return (d2 - d1).days if d1 and d2 else None

def store_in_db(collection, text, metadata):
    """Store document in ChromaDB"""
    doc_id = hashlib.md5(f"{text}_{datetime.now()}".encode()).hexdigest()
    db[collection].add(documents=[text], metadatas=[metadata], ids=[doc_id])

def search_db(collection, query, n=5):
    """Query ChromaDB"""
    try:
        results = db[collection].query(query_texts=[query], n_results=n)
        return [{"text": doc, "meta": meta} 
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
    except:
        return []

def get_location(place):
    """Get coordinates"""
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", 
                     params={"name": place, "count": 1}, timeout=10).json()
    if "results" in r:
        d = r["results"][0]
        return {"name": d["name"], "country": d.get("country", ""), 
                "lat": d["latitude"], "lon": d["longitude"]}
    return None

def get_weather(lat, lon):
    """Get weather"""
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude": lat, "longitude": lon, "current_weather": True},
                     timeout=10).json()
    w = r.get("current_weather", {})
    return {"temp": w.get("temperature"), "wind": w.get("windspeed")}

def get_attractions(lat, lon, city, serper_key):
    """Find attractions"""
    query = f"[out:json][timeout:15];(node['tourism'='attraction']({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1}););out 8;"
    r = requests.post("http://overpass-api.de/api/interpreter", data=query, timeout=20).json()
    
    attractions = []
    for elem in r.get("elements", []):
        if "tags" in elem and "name" in elem["tags"]:
            name = elem["tags"]["name"]
            attractions.append({"name": name, "type": elem["tags"].get("tourism", "attraction")})
            
            store_in_db("attractions", 
                       f"Attraction: {name} in {city}, Type: {elem['tags'].get('tourism')}",
                       {"name": name, "city": city})
    
    return attractions

def generate_plan(dest_info, days, budget, budget_type, ai_model):
    """Generate trip plan with RAG"""
    rag_results = search_db("attractions", f"attractions in {dest_info['name']}", n=3)
    rag_context = "\n".join([r["text"] for r in rag_results]) if rag_results else ""
    
    attractions = ", ".join([a["name"] for a in dest_info.get("attractions", [])[:5]])
    budget_info = ""
    if budget:
        daily = budget/days if budget_type == "Total Budget" else budget
        budget_info = f"\nBudget: ${budget:,.0f} ({budget_type}) = ${daily:,.0f}/day"
    
    prompt = f"""Create a {days}-day travel plan for {dest_info['name']}, {dest_info['country']}.

Previous Knowledge: {rag_context}

Details:
- Weather: {dest_info.get('temp', 'N/A')}°C
- Attractions: {attractions}{budget_info}

Create day-by-day itinerary with activities and budget tips."""
    
    response = ai_model(prompt)
    return response.content if hasattr(response, 'content') else str(response)

# ==================== AI SETUP ====================

def setup_ai(provider, key, model):
    """Setup AI model"""
    if not key:
        return None
    
    if provider == "Google Gemini":
        import google.generativeai as genai
        genai.configure(api_key=key)
        model_obj = genai.GenerativeModel(model)
        return lambda p: type('R', (), {'content': model_obj.generate_content(p).text})()
    else:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        return lambda p: type('R', (), {'content': client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": p}]
        ).choices[0].message.content})()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    provider = st.selectbox("🤖 AI Provider", ["Google Gemini", "OpenAI"])
    
    if provider == "Google Gemini":
        ai_key = st.text_input("🔑 Gemini API Key", type="password", 
                              help="Get free key at ai.google.dev")
        model = st.selectbox("📦 Model", ["gemini-2.0-flash-exp", "gemini-1.5-pro"])
    else:
        ai_key = st.text_input("🔑 OpenAI API Key", type="password")
        model = st.selectbox("📦 Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
    
    serper_key = st.text_input("🔍 Serper API Key", type="password", 
                              help="Optional - for enhanced search")
    
    st.markdown("---")
    st.markdown("## 🗄️ Vector Database")
    
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📍 Attractions", db["attractions"].count())
        with col2:
            st.metric("🎫 Tickets", db["tickets"].count())
        
        if st.button("🗑️ Clear Database", help="Delete all stored data"):
            for col in db.values():
                try:
                    col.delete()
                except:
                    pass
            st.success("Database cleared!")
            st.rerun()
    except:
        st.info("Database empty")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Links")
    st.markdown("- [Get Gemini API](https://ai.google.dev)")
    st.markdown("- [Get Serper API](https://serper.dev)")
    st.markdown("- [Documentation](https://docs.streamlit.io)")

# ==================== MAIN CONTENT ====================

# Progress indicator
st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <span class='step-indicator'>1️⃣ Upload Tickets</span>
        <span class='step-indicator'>2️⃣ Enter Details</span>
        <span class='step-indicator'>3️⃣ Get Itinerary</span>
    </div>
""", unsafe_allow_html=True)

# Ticket Upload Section
with st.container():
    st.markdown("### 🎫 Smart Ticket Parser")
    st.markdown("*Upload your tickets for automatic destination and date extraction*")
    
    col1, col2 = st.columns(2)
    
    tickets = {"outbound": None, "return": None}
    
    with col1:
        st.markdown("**🛫 Outbound Ticket**")
        out_file = st.file_uploader("", type=['pdf', 'png', 'jpg'], key="out", label_visibility="collapsed")
        if out_file:
            with st.spinner("🔍 Analyzing ticket..."):
                tickets["outbound"] = parse_ticket(out_file)
                store_in_db("tickets", str(tickets["outbound"]), tickets["outbound"])
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success("✅ Ticket parsed successfully!")
            st.json(tickets["outbound"])
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🛬 Return Ticket**")
        ret_file = st.file_uploader("", type=['pdf', 'png', 'jpg'], key="ret", label_visibility="collapsed")
        if ret_file:
            with st.spinner("🔍 Analyzing ticket..."):
                tickets["return"] = parse_ticket(ret_file)
                store_in_db("tickets", str(tickets["return"]), tickets["return"])
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success("✅ Ticket parsed successfully!")
            st.json(tickets["return"])
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-calculate days
    auto_days = None
    if tickets["outbound"] and tickets["return"]:
        if tickets["outbound"]["date"] and tickets["return"]["date"]:
            auto_days = calc_days(tickets["outbound"]["date"], tickets["return"]["date"])
            if auto_days:
                st.markdown(f"""
                    <div class='success-box'>
                        🗓️ Trip Duration Auto-Calculated: {auto_days} days
                    </div>
                """, unsafe_allow_html=True)

st.markdown("---")

# Trip Planning Section
with st.container():
    st.markdown("### ✈️ Plan Your Perfect Trip")
    
    default_dest = tickets["outbound"]["destination"] if tickets["outbound"] and tickets["outbound"]["destination"] else ""
    destination = st.text_input("🌍 Where do you want to go?", 
                               value=default_dest, 
                               placeholder="e.g., Paris, Tokyo, New York",
                               help="Enter your dream destination")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days = st.number_input("📅 Trip Duration (days)", 1, 30, auto_days or 7)
    
    with col2:
        budget = st.number_input("💰 Budget (USD)", 0, step=100, value=1000)
    
    with col3:
        budget_type = st.selectbox("💳 Budget Type", ["Total Budget", "Daily Budget"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate Plan Button
    if st.button("🚀 Generate My Travel Plan", type="primary"):
        if not destination or not ai_key:
            st.error("⚠️ Please enter a destination and add your API key in the sidebar!")
        else:
            ai_model = setup_ai(provider, ai_key, model)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎨 Creating your personalized itinerary..."):
                # Step 1: Get location
                status_text.text("📍 Finding location...")
                progress_bar.progress(20)
                loc = get_location(destination)
                if not loc:
                    st.error("❌ Location not found! Please try a different destination.")
                    st.stop()
                
                # Step 2: Get weather
                status_text.text("⛅ Fetching weather data...")
                progress_bar.progress(40)
                weather = get_weather(loc["lat"], loc["lon"])
                
                # Step 3: Get attractions
                status_text.text("🏛️ Discovering attractions...")
                progress_bar.progress(60)
                attractions = get_attractions(loc["lat"], loc["lon"], loc["name"], serper_key)
                
                # Step 4: Generate plan
                status_text.text("🤖 Generating personalized itinerary...")
                progress_bar.progress(80)
                dest_info = {**loc, **weather, "attractions": attractions}
                plan = generate_plan(dest_info, days, budget, budget_type, ai_model)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Success Message
            st.markdown(f"""
                <div class='success-box'>
                    🎉 Your personalized {days}-day trip to {loc['name']} is ready!
                </div>
            """, unsafe_allow_html=True)
            
            # Location Info
            st.markdown("### 📍 Destination Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🌍</h3>
                        <h4>{loc['name']}</h4>
                        <p>{loc['country']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🌡️</h3>
                        <h4>{weather['temp']}°C</h4>
                        <p>Temperature</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>💨</h3>
                        <h4>{weather['wind']} km/h</h4>
                        <p>Wind Speed</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🏛️</h3>
                        <h4>{len(attractions)}</h4>
                        <p>Attractions</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Attractions
            if attractions:
                st.markdown("### 🏛️ Top Attractions")
                cols = st.columns(2)
                for i, a in enumerate(attractions[:6]):
                    with cols[i % 2]:
                        st.markdown(f"""
                            <div class='attraction-card'>
                                <h4>📌 {a['name']}</h4>
                                <p style='color: #666;'>Type: {a['type'].title()}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Budget Breakdown
            if budget:
                st.markdown("### 💰 Budget Breakdown")
                col1, col2, col3, col4 = st.columns(4)
                
                if budget_type == "Total Budget":
                    daily = budget / days
                    col1.metric("💵 Total Budget", f"${budget:,.0f}")
                    col2.metric("📅 Daily Budget", f"${daily:,.0f}")
                    col3.metric("🍽️ Per Meal", f"${daily/3:,.0f}")
                    col4.metric("🛏️ Accommodation", f"${daily*0.4:,.0f}")
                else:
                    total = budget * days
                    col1.metric("📅 Daily Budget", f"${budget:,.0f}")
                    col2.metric("💵 Total Budget", f"${total:,.0f}")
                    col3.metric("🍽️ Per Meal", f"${budget/3:,.0f}")
                    col4.metric("🛏️ Accommodation", f"${budget*0.4:,.0f}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Itinerary
            st.markdown("### 📅 Your Personalized Itinerary")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(plan)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download Button
            st.download_button(
                "📥 Download Complete Itinerary",
                plan,
                file_name=f"trip_{destination}_{datetime.now():%Y%m%d}.txt",
                mime="text/plain"
            )

st.markdown("---")

# RAG Query Tool
with st.expander("🔍 Search Vector Database", expanded=False):
    st.markdown("*Query stored travel information using semantic search*")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("", placeholder="e.g., best attractions in Paris", label_visibility="collapsed")
    with col2:
        collection = st.selectbox("", ["attractions", "tickets", "reviews"], label_visibility="collapsed")
    
    if st.button("🔍 Search Database"):
        if query:
            with st.spinner("Searching..."):
                results = search_db(collection, query)
                if results:
                    st.success(f"Found {len(results)} results!")
                    for i, r in enumerate(results, 1):
                        st.markdown(f"**Result {i}:**")
                        st.info(r["text"])
                else:
                    st.warning("No results found")
        else:
            st.warning("Please enter a search query")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🤖 Powered by ChromaDB RAG</strong></p>
        <p>🗄️ Persistent Vector Storage • 🎫 Smart Ticket Parser • 💰 Budget Planning</p>
        <p style='font-size: 12px;'>Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
