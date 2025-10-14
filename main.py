"""
Smart Travel Planner - Clean & Readable UI
ChromaDB RAG + Ticket Parser + Budget Planning
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
    page_title="Universal AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Universal AI Travel Planner")
st.caption("Smart ticket parser • Budget planning • ChromaDB vector database")

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
    st.header("⚙️ Configuration")
    
    provider = st.selectbox("AI Provider", ["Google Gemini", "OpenAI"])
    
    if provider == "Google Gemini":
        ai_key = st.text_input("Gemini API Key", type="password")
        model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    else:
        ai_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
    
    serper_key = st.text_input("Serper API Key (optional)", type="password")
    
    st.divider()
    st.subheader("🗄️ Vector Database")
    
    try:
        st.metric("Attractions Stored", db["attractions"].count())
        st.metric("Tickets Stored", db["tickets"].count())
        
        if st.button("Clear Database"):
            for col in db.values():
                try:
                    col.delete()
                except:
                    pass
            st.success("Database cleared!")
            st.rerun()
    except:
        st.info("Database is empty")

# ==================== MAIN CONTENT ====================

st.header("🎫 Step 1: Upload Tickets (Optional)")
st.write("Upload your tickets for automatic destination and date extraction")

col1, col2 = st.columns(2)

tickets = {"outbound": None, "return": None}

with col1:
    st.subheader("Outbound Ticket")
    out_file = st.file_uploader("Upload outbound ticket", type=['pdf', 'png', 'jpg'], key="out")
    if out_file:
        with st.spinner("Parsing ticket..."):
            tickets["outbound"] = parse_ticket(out_file)
            store_in_db("tickets", str(tickets["outbound"]), tickets["outbound"])
        st.success("✅ Ticket parsed!")
        st.json(tickets["outbound"])

with col2:
    st.subheader("Return Ticket")
    ret_file = st.file_uploader("Upload return ticket", type=['pdf', 'png', 'jpg'], key="ret")
    if ret_file:
        with st.spinner("Parsing ticket..."):
            tickets["return"] = parse_ticket(ret_file)
            store_in_db("tickets", str(tickets["return"]), tickets["return"])
        st.success("✅ Ticket parsed!")
        st.json(tickets["return"])

# Auto-calculate days
auto_days = None
if tickets["outbound"] and tickets["return"]:
    if tickets["outbound"]["date"] and tickets["return"]["date"]:
        auto_days = calc_days(tickets["outbound"]["date"], tickets["return"]["date"])
        if auto_days:
            st.info(f"🗓️ Auto-calculated trip duration: **{auto_days} days**")

st.divider()

# Trip Planning
st.header("✈️ Step 2: Enter Trip Details")

default_dest = tickets["outbound"]["destination"] if tickets["outbound"] and tickets["outbound"]["destination"] else ""
destination = st.text_input("Destination", value=default_dest, placeholder="e.g., Paris, Tokyo, New York")

col1, col2, col3 = st.columns(3)

with col1:
    days = st.number_input("Trip Days", 1, 30, auto_days or 7)

with col2:
    budget = st.number_input("Budget ($)", 0, step=100, value=1000)

with col3:
    budget_type = st.selectbox("Budget Type", ["Total Budget", "Daily Budget"])

st.divider()

# Generate Plan
st.header("🚀 Step 3: Generate Itinerary")

if st.button("Generate Travel Plan", type="primary", use_container_width=True):
    if not destination or not ai_key:
        st.error("⚠️ Please enter a destination and add your API key in the sidebar!")
    else:
        ai_model = setup_ai(provider, ai_key, model)
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        with st.spinner("Creating your itinerary..."):
            # Get location
            status.text("📍 Finding location...")
            progress_bar.progress(20)
            loc = get_location(destination)
            if not loc:
                st.error("❌ Location not found!")
                st.stop()
            
            # Get weather
            status.text("⛅ Fetching weather...")
            progress_bar.progress(40)
            weather = get_weather(loc["lat"], loc["lon"])
            
            # Get attractions
            status.text("🏛️ Finding attractions...")
            progress_bar.progress(60)
            attractions = get_attractions(loc["lat"], loc["lon"], loc["name"], serper_key)
            
            # Generate plan
            status.text("🤖 Generating itinerary...")
            progress_bar.progress(80)
            dest_info = {**loc, **weather, "attractions": attractions}
            plan = generate_plan(dest_info, days, budget, budget_type, ai_model)
            
            progress_bar.progress(100)
            status.text("✅ Complete!")
        
        st.success(f"🎉 Your {days}-day trip to {loc['name']} is ready!")
        
        # Location Info
        st.subheader("📍 Destination Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Location", loc['name'])
        with col2:
            st.metric("Country", loc['country'])
        with col3:
            st.metric("Temperature", f"{weather['temp']}°C")
        with col4:
            st.metric("Wind Speed", f"{weather['wind']} km/h")
        
        # Attractions
        if attractions:
            st.subheader("🏛️ Top Attractions")
            for i, a in enumerate(attractions[:8], 1):
                st.write(f"**{i}. {a['name']}** - {a['type'].title()}")
        
        # Budget Breakdown
        if budget:
            st.subheader("💰 Budget Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            
            if budget_type == "Total Budget":
                daily = budget / days
                with col1:
                    st.metric("Total Budget", f"${budget:,.0f}")
                with col2:
                    st.metric("Daily Budget", f"${daily:,.0f}")
                with col3:
                    st.metric("Per Meal (est)", f"${daily/3:,.0f}")
                with col4:
                    st.metric("Accommodation (est)", f"${daily*0.4:,.0f}")
            else:
                total = budget * days
                with col1:
                    st.metric("Daily Budget", f"${budget:,.0f}")
                with col2:
                    st.metric("Total Budget", f"${total:,.0f}")
                with col3:
                    st.metric("Per Meal (est)", f"${budget/3:,.0f}")
                with col4:
                    st.metric("Accommodation (est)", f"${budget*0.4:,.0f}")
        
        # Itinerary
        st.subheader("📅 Your Personalized Itinerary")
        st.write(plan)
        
        # Download
        st.download_button(
            "📥 Download Itinerary",
            plan,
            file_name=f"trip_{destination}_{datetime.now():%Y%m%d}.txt",
            use_container_width=True
        )

st.divider()

# RAG Query Tool
with st.expander("🔍 Search Vector Database"):
    st.write("Query stored travel information using semantic search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search query", placeholder="e.g., best attractions in Paris")
    with col2:
        collection = st.selectbox("Collection", ["attractions", "tickets", "reviews"])
    
    if st.button("Search"):
        if query:
            results = search_db(collection, query)
            if results:
                st.success(f"Found {len(results)} results")
                for i, r in enumerate(results, 1):
                    st.write(f"**Result {i}:**")
                    st.info(r["text"])
            else:
                st.warning("No results found")

# Footer
st.divider()
st.caption("🤖 Powered by ChromaDB RAG • 🎫 Smart Ticket Parser • 💰 Budget Planning")

# Help
with st.expander("❓ Help & Documentation"):
    st.markdown("""
    ### Installation:
    ```bash
    pip install -r requirements.txt
    ```
    
    ### For OCR (Image Tickets):
    - **Mac:** `brew install tesseract`
    - **Linux:** `sudo apt-get install tesseract-ocr`
    - **Windows:** Download from GitHub
    
    ### Usage:
    1. **Upload tickets** (optional) → Auto-fills destination & days
    2. **Enter trip details** manually if no tickets
    3. **Add API key** in sidebar (Gemini is free at ai.google.dev)
    4. **Click Generate** to create your itinerary
    
    ### ChromaDB Features:
    - Stores all attractions, tickets, and reviews
    - Enables semantic search across collections
    - RAG enhances AI responses with stored knowledge
    - Data persists in `./chroma_db/` folder
    
    ### Deployment:
    - Works on Streamlit Cloud
    - Add `requirements.txt` to your repo
    - Optionally add `packages.txt` with tesseract-ocr for OCR support
    """)
