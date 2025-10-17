"""
Smart Travel Planner - Clean & Readable UI
FAISS Vector DB + Ticket Parser + Budget Planning + Hotels & Restaurants
"""
import streamlit as st
import requests
from datetime import datetime
import re
import PyPDF2
from PIL import Image
import pytesseract
import faiss
import numpy as np
import pickle
import os
import random

# Page config
st.set_page_config(
    page_title="Universal AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for better readability
st.markdown("""
<style>
    /* Better text contrast and spacing */
    .stMarkdown, .stText {
        line-height: 1.6;
    }
    
    /* Card-like containers */
    .element-container {
        padding: 0.5rem 0;
    }
    
    /* Better button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: 2px solid rgba(49, 51, 63, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        border-color: rgba(49, 51, 63, 0.4);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Better metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Improve dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid rgba(49, 51, 63, 0.1);
    }
    
    /* Card styling for containers */
    .stContainer {
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(49, 51, 63, 0.1);
        background: rgba(255, 255, 255, 0.02);
        margin-bottom: 1rem;
    }
    
    /* Better image styling */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
    }
    
    /* Header styling */
    h1 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid rgba(49, 51, 63, 0.1);
    }
    
    h3 {
        font-weight: 600;
        color: rgba(49, 51, 63, 0.9);
    }
    
    /* Better caption styling */
    .caption, [data-testid="stCaptionContainer"] {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 0.25rem;
    }
    
    /* Info/Success boxes with better contrast */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
        background: rgba(59, 130, 246, 0.05);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(59, 130, 246, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background: rgba(99, 102, 241, 0.05);
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Better spacing for columns */
    [data-testid="column"] {
        padding: 0 0.75rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        border-radius: 10px;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(49, 51, 63, 0.2);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 6px;
        border: 2px solid rgba(49, 51, 63, 0.2);
        padding: 0.5rem;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: rgba(49, 51, 63, 0.4);
        box-shadow: 0 0 0 2px rgba(49, 51, 63, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("✈️ Universal AI Travel Planner")
st.caption("Smart ticket parser • Budget planning • Hotels & Restaurants • FAISS vector database")

# ==================== FAISS SETUP ====================

@st.cache_resource
def init_db():
    """Initialize FAISS database"""
    os.makedirs("./faiss_db", exist_ok=True)
    
    db = {
        "index": None,
        "documents": [],
        "dimension": 384  # Simple embedding dimension
    }
    
    # Try to load existing database
    if os.path.exists("./faiss_db/index.faiss"):
        try:
            db["index"] = faiss.read_index("./faiss_db/index.faiss")
            with open("./faiss_db/documents.pkl", "rb") as f:
                db["documents"] = pickle.load(f)
        except:
            db["index"] = faiss.IndexFlatL2(db["dimension"])
    else:
        db["index"] = faiss.IndexFlatL2(db["dimension"])
    
    return db

db = init_db()

# ==================== HELPER FUNCTIONS ====================

def simple_embed(text):
    """Simple text embedding using hash-based approach"""
    # Simple deterministic embedding
    words = text.lower().split()
    embedding = np.zeros(384)
    for i, word in enumerate(words[:50]):
        hash_val = hash(word)
        idx = abs(hash_val) % 384
        embedding[idx] += 1.0 / (i + 1)
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype('float32')

def store_in_db(text, metadata):
    """Store document in FAISS"""
    embedding = simple_embed(text)
    db["index"].add(np.array([embedding]))
    db["documents"].append({"text": text, "metadata": metadata})
    
    # Save to disk
    faiss.write_index(db["index"], "./faiss_db/index.faiss")
    with open("./faiss_db/documents.pkl", "wb") as f:
        pickle.dump(db["documents"], f)

def search_db(query, n=5):
    """Query FAISS database"""
    if db["index"].ntotal == 0:
        return []
    
    query_embedding = simple_embed(query)
    distances, indices = db["index"].search(np.array([query_embedding]), min(n, db["index"].ntotal))
    
    results = []
    for idx in indices[0]:
        if idx < len(db["documents"]):
            results.append(db["documents"][idx])
    
    return results

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

def get_attraction_image(attraction_name, city):
    """Get image URL for attraction using Pexels API with fallback"""
    try:
        # Generate a seed from attraction name for consistent but varied images
        seed = abs(hash(attraction_name)) % 1000
        
        # Smaller image size: 200x150
        picsum_url = f"https://picsum.photos/seed/{seed}/200/150"
        
        return picsum_url
        
    except:
        # Ultimate fallback
        return "https://picsum.photos/200/150?random"

def get_hotel_image(hotel_name):
    """Get image for hotels"""
    seed = abs(hash(hotel_name)) % 1000
    return f"https://picsum.photos/seed/hotel{seed}/200/150"

def get_restaurant_image(restaurant_name):
    """Get image for restaurants"""
    seed = abs(hash(restaurant_name)) % 1000
    return f"https://picsum.photos/seed/food{seed}/200/150"

def get_attractions(lat, lon, city):
    """Find attractions"""
    query = f"[out:json][timeout:15];(node['tourism'='attraction']({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1}););out 8;"
    r = requests.post("http://overpass-api.de/api/interpreter", data=query, timeout=20).json()
    
    attractions = []
    for elem in r.get("elements", []):
        if "tags" in elem and "name" in elem["tags"]:
            name = elem["tags"]["name"]
            rating = round(random.uniform(3.7, 4.5), 1)
            image_url = get_attraction_image(name, city)
            
            attractions.append({
                "name": name, 
                "type": elem["tags"].get("tourism", "attraction"),
                "rating": rating,
                "image": image_url
            })
            
            store_in_db(
                f"Attraction: {name} in {city}, Type: {elem['tags'].get('tourism')}, Rating: {rating}",
                {"name": name, "city": city, "type": "attraction", "rating": rating}
            )
    
    return attractions

def get_hotels(lat, lon, city):
    """Find hotels nearby"""
    query = f"[out:json][timeout:15];(node['tourism'='hotel']({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05});way['tourism'='hotel']({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05}););out 10;"
    try:
        r = requests.post("http://overpass-api.de/api/interpreter", data=query, timeout=20).json()
        
        hotels = []
        for elem in r.get("elements", []):
            if "tags" in elem and "name" in elem["tags"]:
                name = elem["tags"]["name"]
                stars = elem["tags"].get("stars", str(random.randint(3, 5)))
                rating = round(random.uniform(3.7, 4.5), 1)
                image_url = get_hotel_image(name)
                
                hotel_data = {
                    "name": name,
                    "stars": stars,
                    "rating": rating,
                    "address": elem["tags"].get("addr:street", "Address not available"),
                    "phone": elem["tags"].get("phone", "N/A"),
                    "website": elem["tags"].get("website", "N/A"),
                    "image": image_url
                }
                hotels.append(hotel_data)
                
                store_in_db(
                    f"Hotel: {name} in {city}, Stars: {stars}, Rating: {rating}",
                    {"name": name, "city": city, "type": "hotel", "stars": stars, "rating": rating}
                )
        
        return hotels
    except:
        return []

def get_restaurants(lat, lon, city):
    """Find restaurants nearby"""
    query = f"[out:json][timeout:15];(node['amenity'='restaurant']({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05});way['amenity'='restaurant']({lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05}););out 12;"
    try:
        r = requests.post("http://overpass-api.de/api/interpreter", data=query, timeout=20).json()
        
        restaurants = []
        for elem in r.get("elements", []):
            if "tags" in elem and "name" in elem["tags"]:
                name = elem["tags"]["name"]
                cuisine = elem["tags"].get("cuisine", "International").title()
                rating = round(random.uniform(3.7, 4.5), 1)
                image_url = get_restaurant_image(name)
                
                rest_data = {
                    "name": name,
                    "cuisine": cuisine,
                    "rating": rating,
                    "address": elem["tags"].get("addr:street", "Address not available"),
                    "phone": elem["tags"].get("phone", "N/A"),
                    "website": elem["tags"].get("website", "N/A"),
                    "image": image_url
                }
                restaurants.append(rest_data)
                
                store_in_db(
                    f"Restaurant: {name} in {city}, Cuisine: {cuisine}, Rating: {rating}",
                    {"name": name, "city": city, "type": "restaurant", "cuisine": cuisine, "rating": rating}
                )
        
        return restaurants
    except:
        return []

def generate_plan(dest_info, days, budget, budget_type, ai_model):
    """Generate trip plan with RAG"""
    rag_results = search_db(f"attractions in {dest_info['name']}", n=3)
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

Create day-by-day itinerary with activities and budget tips.

IMPORTANT FORMATTING RULES:
1. Use plain text only - NO markdown symbols like *, **, or ***
2. For currency, write: "USD 650" or "$650" (no asterisks)
3. For emphasis, use CAPS or write normally
4. Structure each day clearly with "Day 1:", "Day 2:", etc.
5. Use simple bullet points with "-" or numbers
6. Keep all text clean and readable without any special formatting characters"""
    
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
    
    st.divider()
    st.subheader("🗄️ FAISS Database")
    
    st.metric("Documents Stored", db["index"].ntotal)
    
    if st.button("Clear Database"):
        db["index"].reset()
        db["documents"] = []
        faiss.write_index(db["index"], "./faiss_db/index.faiss")
        with open("./faiss_db/documents.pkl", "wb") as f:
            pickle.dump(db["documents"], f)
        st.success("Database cleared!")
        st.rerun()

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
            store_in_db(str(tickets["outbound"]), tickets["outbound"])
        st.success("✅ Ticket parsed!")
        st.json(tickets["outbound"])

with col2:
    st.subheader("Return Ticket")
    ret_file = st.file_uploader("Upload return ticket", type=['pdf', 'png', 'jpg'], key="ret")
    if ret_file:
        with st.spinner("Parsing ticket..."):
            tickets["return"] = parse_ticket(ret_file)
            store_in_db(str(tickets["return"]), tickets["return"])
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
            progress_bar.progress(50)
            attractions = get_attractions(loc["lat"], loc["lon"], loc["name"])
            
            # Get hotels
            status.text("🏨 Finding hotels...")
            progress_bar.progress(65)
            hotels = get_hotels(loc["lat"], loc["lon"], loc["name"])
            
            # Get restaurants
            status.text("🍽️ Finding restaurants...")
            progress_bar.progress(75)
            restaurants = get_restaurants(loc["lat"], loc["lon"], loc["name"])
            
            # Generate plan
            status.text("🤖 Generating itinerary...")
            progress_bar.progress(90)
            dest_info = {**loc, **weather, "attractions": attractions, "hotels": hotels, "restaurants": restaurants}
            plan = generate_plan(dest_info, days, budget, budget_type, ai_model)
            
            progress_bar.progress(100)
            status.text("✅ Complete!")
        
        st.success(f"🎉 Your {days}-day trip to {loc['name']} is ready!")
        
        # Location Info with enhanced cards
        st.subheader("📍 Destination Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 197, 253, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid rgba(59, 130, 246, 0.2);">
                <h3 style="margin: 0; color: #e5e7eb; font-size: 1.1rem;">📍 Location</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 700; color: #93c5fd;">{}</p>
            </div>
            """.format(loc['name']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(110, 231, 183, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.2);">
                <h3 style="margin: 0; color: #e5e7eb; font-size: 1.1rem;">🌍 Country</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 700; color: #6ee7b7;">{}</p>
            </div>
            """.format(loc['country']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(251, 191, 36, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid rgba(249, 115, 22, 0.2);">
                <h3 style="margin: 0; color: #e5e7eb; font-size: 1.1rem;">🌡️ Temperature</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 700; color: #fbbf24;">{}°C</p>
            </div>
            """.format(weather['temp']), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(196, 181, 253, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid rgba(139, 92, 246, 0.2);">
                <h3 style="margin: 0; color: #e5e7eb; font-size: 1.1rem;">💨 Wind Speed</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 700; color: #c4b5fd;">{} km/h</p>
            </div>
            """.format(weather['wind']), unsafe_allow_html=True)
        
        st.markdown("")
        
        # Attractions Carousel
        if attractions:
            st.subheader("🏛️ Top Attractions")
            st.write("")
            
            # Create tabs for carousel-like experience
            attraction_tabs = st.tabs([f"📍 {i+1}" for i in range(min(len(attractions), 9))])
            
            for i, (tab, attraction) in enumerate(zip(attraction_tabs, attractions[:9])):
                with tab:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(attraction['image'], use_container_width=True)
                    with col2:
                        st.markdown(f"### {attraction['name']}")
                        st.markdown(f"**⭐ {attraction['rating']}**")
                        st.markdown(f"**Type:** {attraction['type'].title()}")
                        st.markdown(f"**Rank:** #{i+1}")
            
            st.markdown("")
        
        # Hotels & Restaurants Side by Side
        col_left, col_right = st.columns(2)
        
        with col_left:
            if hotels:
                st.subheader("🏨 Recommended Hotels")
                st.write("")
                
                for i, hotel in enumerate(hotels[:6], 1):
                    with st.container():
                        # Card design
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(168, 85, 247, 0.05) 100%); 
                                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1; margin-bottom: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #f9fafb;">{i}. {hotel['name']}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        subcol1, subcol2 = st.columns([1, 3])
                        with subcol1:
                            st.image(hotel['image'], use_container_width=True)
                        with subcol2:
                            st.markdown(f"**⭐ {hotel['rating']}** • {hotel['stars']} ⭐")
                            st.markdown(f"📍 {hotel['address']}")
                            if hotel['phone'] != 'N/A':
                                st.markdown(f"📞 {hotel['phone']}")
                            if hotel['website'] != 'N/A':
                                st.markdown(f"🌐 [Website]({hotel['website']})")
                        
                        st.markdown("")
        
        with col_right:
            if restaurants:
                st.subheader("🍽️ Nearby Restaurants")
                st.write("")
                
                for i, rest in enumerate(restaurants[:6], 1):
                    with st.container():
                        # Card design
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.05) 0%, rgba(251, 146, 60, 0.05) 100%); 
                                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ec4899; margin-bottom: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #f9fafb;">{i}. {rest['name']}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        subcol1, subcol2 = st.columns([1, 3])
                        with subcol1:
                            st.image(rest['image'], use_container_width=True)
                        with subcol2:
                            st.markdown(f"**⭐ {rest['rating']}** • 🍴 {rest['cuisine']}")
                            st.markdown(f"📍 {rest['address']}")
                            if rest['phone'] != 'N/A':
                                st.markdown(f"📞 {rest['phone']}")
                            if rest['website'] != 'N/A':
                                st.markdown(f"🌐 [Website]({rest['website']})")
                        
                        st.markdown("")
        
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
        
        # Itinerary with enhanced styling
        st.subheader("📅 Your Personalized Itinerary")
        
        # Create an attractive itinerary box
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(16, 185, 129, 0.08) 100%); 
                    padding: 2rem; border-radius: 12px; border: 2px solid rgba(59, 130, 246, 0.3); margin: 1rem 0;">
        """, unsafe_allow_html=True)
        
        # Clean the plan text from markdown artifacts
        clean_plan = plan.replace('**', '').replace('*', '').replace('~', '')
        
        # Parse the plan into days if possible
        plan_lines = clean_plan.split('\n')
        current_day = None
        day_content = []
        
        for line in plan_lines:
            if line.strip():
                # Check if line is a day header
                if 'day' in line.lower() and any(char.isdigit() for char in line):
                    if current_day and day_content:
                        # Display previous day
                        with st.expander(f"📆 {current_day}", expanded=True):
                            st.markdown('\n'.join(day_content))
                    current_day = line.strip()
                    day_content = []
                else:
                    day_content.append(line)
        
        # Display last day
        if current_day and day_content:
            with st.expander(f"📆 {current_day}", expanded=True):
                st.markdown('\n'.join(day_content))
        
        # If no days detected, show plain text
        if not current_day:
            st.markdown(clean_plan)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Download with better styling
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                "📥 Download Complete Itinerary",
                plan,
                file_name=f"trip_{destination}_{datetime.now():%Y%m%d}.txt",
                use_container_width=True
            )

st.divider()

# RAG Query Tool
with st.expander("🔍 Search Vector Database"):
    st.write("Query stored travel information using FAISS semantic search")
    
    query = st.text_input("Search query", placeholder="e.g., best hotels in Paris")
    
    if st.button("Search"):
        if query:
            results = search_db(query, n=5)
            if results:
                st.success(f"Found {len(results)} results")
                for i, r in enumerate(results, 1):
                    st.write(f"**Result {i}:**")
                    st.info(r["text"])
            else:
                st.warning("No results found")

# Footer
st.divider()
st.caption("🤖 Powered by FAISS • 🎫 Smart Ticket Parser • 🏨 Hotel Recommendations • 🍽️ Restaurant Finder")

# Help
with st.expander("❓ Help & Documentation"):
    st.markdown("""
    ### Installation:
    ```bash
    pip install streamlit requests PyPDF2 Pillow pytesseract faiss-cpu numpy
    pip install google-generativeai openai
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
    
    ### New Features:
    - **🗄️ FAISS Database:** Fast similarity search (replaced ChromaDB)
    - **⭐ Ratings:** Random ratings between 3.7-4.5 for all places
    - **🏨 Hotel Recommendations:** With stars and ratings
    - **🍽️ Restaurant Finder:** With cuisine types and ratings
    - **📍 Location-based Search:** Uses OpenStreetMap data
    
    ### FAISS Features:
    - Lightweight and fast vector search
    - Simple hash-based embeddings
    - Stores all data in `./faiss_db/` folder
    - Persistent storage across sessions
    
    ### Deployment:
    - Works on Streamlit Cloud
    - Lighter than ChromaDB
    - No complex dependencies
    
    ### API Keys:
    - **Gemini (Free):** Get at ai.google.dev
    - **OpenAI:** Get at platform.openai.com
    """)
