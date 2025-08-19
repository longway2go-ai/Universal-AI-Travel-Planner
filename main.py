# streamlit_app.py
"""
Universal Travel Planner - Enhanced Streamlit Web App with Auto-Geocoding & Nearby Places Discovery
Deploy with: streamlit run streamlit_app.py
"""
import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import requests
import re
from typing import Dict, List, Tuple, Optional

# Set page config
st.set_page_config(
    page_title="AI Travel Planner - Discover Amazing Destinations",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2E86AB, #A23B72, #F18F01);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(45deg, #2E86AB, #A23B72);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #A23B72, #F18F01);
        transform: translateY(-2px);
    }
    .example-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌍 Universal AI Travel Planner 🗺️</h1>', unsafe_allow_html=True)
st.markdown("**Discover amazing destinations worldwide!** Plan your perfect trip with AI-powered recommendations, automatic location discovery, and interactive maps.")

# ==================== HELPER FUNCTIONS ====================

def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame columns by removing spaces and converting to lowercase"""
    df.columns = df.columns.str.strip().str.lower()
    return df

def safe_get_column(df, column_name, default_name=None):
    """Safely get column from DataFrame with case-insensitive matching"""
    col_map = {col.lower().strip(): col for col in df.columns}
    target_col = column_name.lower().strip()
    if target_col in col_map:
        return col_map[target_col]
    elif default_name and default_name in df.columns:
        return default_name
    else:
        raise KeyError(f"Column '{column_name}' not found. Available columns: {list(df.columns)}")

# ==================== GEOCODING FUNCTIONS ====================

def geocode_location_with_fallback(location: str) -> Tuple[Optional[float], Optional[float], str]:
    """Try multiple free geocoding APIs as fallback"""
    if not location or location.strip() == "":
        return None, None, "invalid_input"
    
    location = location.strip()
    
    # Method 1: Open-Meteo (completely free, no key needed)
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": location, "count": 1}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            lat = float(result["latitude"])
            lon = float(result["longitude"])
            return lat, lon, "open-meteo"
    except Exception as e:
        st.warning(f"Open-Meteo geocoding failed: {e}")
    
    # Method 2: Nominatim OpenStreetMap (free, but rate limited)
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": location, "format": "json", "limit": 1}
        headers = {"User-Agent": "Universal Travel Planner"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data and len(data) > 0:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lat, lon, "nominatim"
    except Exception as e:
        st.warning(f"Nominatim geocoding failed: {e}")
    
    return None, None, "failed"

# ==================== NEARBY PLACES DISCOVERY ====================

def find_nearby_places_with_ai(model, location_name: str, lat: float, lon: float, radius_km: int = 50):
    """Use AI to find interesting places near a given location"""
    if model is None:
        return []
    
    prompt = f"""
    Find interesting places to visit near {location_name} (coordinates: {lat:.4f}, {lon:.4f}).
    
    REQUIREMENTS:
    - Find 15 diverse attractions, landmarks, or points of interest
    - Include places within approximately {radius_km}km radius
    - For each place provide: exact name, approximate coordinates (lat, lon), type of attraction
    - Include mix of: historical sites, natural attractions, cultural sites, entertainment venues, restaurants, shopping areas
    - Avoid duplicating the main location: {location_name}
    
    FORMAT RESPONSE AS:
    1. Place Name | Latitude | Longitude | Type | Distance from main location
    2. Place Name | Latitude | Longitude | Type | Distance from main location
    ...
    
    Example:
    1. Central Park | 40.7829 | -73.9654 | Natural Park | 2km
    2. Metropolitan Museum | 40.7794 | -73.9632 | Cultural Museum | 1km
    
    Provide realistic coordinates and distances.
    """
    
    try:
        response = model(prompt)
        content = safe_extract_response_text(response)
        return parse_ai_nearby_places(content, location_name)
    except Exception as e:
        st.warning(f"Could not find nearby places with AI: {e}")
        return []

def parse_ai_nearby_places(ai_response: str, main_location: str):
    """Parse AI response to extract nearby places data"""
    places = []
    
    try:
        lines = ai_response.split('\n')
        for line in lines:
            if '|' in line and any(char.isdigit() for char in line):
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    name = parts[0]
                    name = re.sub(r'^\d+\.\s*', '', name)
                    
                    try:
                        lat = float(parts[1])
                        lon = float(parts[2])
                        place_type = parts[3]
                        
                        if name and lat != 0 and lon != 0:
                            places.append({
                                "name": name,
                                "lat": lat,
                                "lon": lon,
                                "type": place_type,
                                "source": "AI_discovery"
                            })
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        st.warning(f"Error parsing AI response: {e}")
    
    return places[:15]

def find_nearby_places_with_api(lat: float, lon: float, radius_km: int = 50):
    """Find nearby places using free web APIs"""
    places = []
    
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["tourism"~"attraction|museum|monument|castle|gallery|zoo|aquarium|theme_park"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
          node["historic"~"castle|monument|museum|ruins|archaeological_site"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
          node["amenity"~"theatre|cinema|library|restaurant|cafe|bar"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
          node["leisure"~"park|garden|sports_centre|marina|beach_resort"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
          node["natural"~"beach|peak|hot_spring|waterfall"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
        );
        out center meta;
        """
        
        response = requests.post(overpass_url, data=overpass_query, timeout=30)
        data = response.json()
        
        if 'elements' in data:
            for element in data['elements'][:20]:
                if 'tags' in element and 'name' in element['tags']:
                    place_lat = element.get('lat', 0)
                    place_lon = element.get('lon', 0)
                    name = element['tags']['name']
                    
                    tags = element['tags']
                    place_type = tags.get('tourism', tags.get('historic', tags.get('amenity', tags.get('leisure', tags.get('natural', 'attraction')))))
                    
                    if place_lat and place_lon and name:
                        places.append({
                            "name": name,
                            "lat": place_lat,
                            "lon": place_lon,
                            "type": place_type,
                            "source": "OpenStreetMap"
                        })
        
    except Exception as e:
        st.warning(f"Could not fetch nearby places from web API: {e}")
    
    return places[:15]

# ==================== TRAVEL EXAMPLES ====================

TRAVEL_EXAMPLES = {
    "🏛️ Historical Europe": {
        "description": "Explore Europe's rich history and culture",
        "locations": {
            "Rome, Italy": (41.9028, 12.4964),
            "Athens, Greece": (37.9755, 23.7348),
            "Paris, France": (48.8566, 2.3522),
            "Prague, Czech Republic": (50.0755, 14.4378),
            "Vienna, Austria": (48.2082, 16.3738)
        }
    },
    "🏝️ Tropical Paradise": {
        "description": "Relax in stunning tropical destinations",
        "locations": {
            "Bali, Indonesia": (-8.3405, 115.0920),
            "Maldives": (3.2028, 73.2207),
            "Seychelles": (-4.6796, 55.4920),
            "Mauritius": (-20.3484, 57.5522),
            "Santorini, Greece": (36.3932, 25.4615)
        }
    },
    "🏔️ Adventure Mountains": {
        "description": "Conquer peaks and enjoy breathtaking views",
        "locations": {
            "Swiss Alps, Switzerland": (46.5197, 7.9738),
            "Himalayas, Nepal": (28.0000, 84.0000),
            "Rocky Mountains, USA": (40.3428, -105.6836),
            "Andes, Peru": (-13.1631, -72.5450),
            "Patagonia, Argentina": (-50.5025, -73.0061)
        }
    },
    "🏙️ Modern Metropolis": {
        "description": "Experience vibrant city life and culture",
        "locations": {
            "Tokyo, Japan": (35.6762, 139.6503),
            "New York, USA": (40.7128, -74.0060),
            "Dubai, UAE": (25.2048, 55.2708),
            "Singapore": (1.3521, 103.8198),
            "London, UK": (51.5074, -0.1278)
        }
    },
    "🎨 Art & Culture": {
        "description": "Immerse yourself in world-class art and culture",
        "locations": {
            "Florence, Italy": (43.7696, 11.2558),
            "Barcelona, Spain": (41.3851, 2.1734),
            "St. Petersburg, Russia": (59.9311, 30.3609),
            "Cairo, Egypt": (30.0444, 31.2357),
            "Kyoto, Japan": (35.0116, 135.7681)
        }
    },
    "🏖️ Beach Getaway": {
        "description": "Unwind on the world's most beautiful beaches",
        "locations": {
            "Cancun, Mexico": (21.1619, -86.8515),
            "Phuket, Thailand": (7.8804, 98.3923),
            "Gold Coast, Australia": (-28.0167, 153.4000),
            "Ibiza, Spain": (38.9067, 1.4206),
            "Hawaii, USA": (21.3099, -157.8581)
        }
    }
}

# ==================== SIDEBAR CONFIGURATION ====================

st.sidebar.header("🧳 Trip Configuration")

# API Keys setup
with st.sidebar.expander("🔐 API Keys Setup"):
    st.info("Enter your API keys to enable full AI functionality")
    
    ai_provider = st.selectbox(
        "🤖 Choose AI Provider",
        ["OpenAI", "Google Gemini", "Together AI"],
        help="Select your preferred AI model provider"
    )
    
    if ai_provider == "OpenAI":
        ai_key = st.text_input("OpenAI API Key", type="password", help="For GPT models")
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    elif ai_provider == "Google Gemini":
        ai_key = st.text_input("Google AI API Key", type="password", help="For Gemini models")
        model_name = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"])
    else:
        ai_key = st.text_input("Together AI API Key", type="password", help="For open-source models")
        model_name = st.selectbox("Model", ["Qwen/Qwen2.5-Coder-32B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"])

# Trip parameters
trip_days = st.sidebar.slider("Trip Duration (days)", 2, 14, 7)
daily_budget_hours = st.sidebar.slider("Daily Travel Budget (hours)", 2, 12, 6)
budget_range = st.sidebar.selectbox("Budget Range", ["Budget ($)", "Mid-range ($$)", "Luxury ($$$)", "Ultra-luxury ($$$$)"])

# Enhanced origin location selection
st.sidebar.subheader("🌍 Origin Location")
location_method = st.sidebar.radio(
    "Select Origin Method:",
    ["🏙️ Popular Cities", "📍 Enter Location", "🗺️ Exact Coordinates"]
)

POPULAR_CITIES = {
    "London, UK": (51.5074, -0.1278),
    "New York, USA": (40.7128, -74.0060),
    "Paris, France": (48.8566, 2.3522),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Mumbai, India": (19.0760, 72.8777),
    "Delhi, India": (28.6139, 77.2090),
    "Singapore": (1.3521, 103.8198),
    "Dubai, UAE": (25.2048, 55.2708),
    "Berlin, Germany": (52.5200, 13.4050),
    "Rome, Italy": (41.9028, 12.4964),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Los Angeles, USA": (34.0522, -118.2437),
    "Toronto, Canada": (43.6532, -79.3832),
    "São Paulo, Brazil": (-23.5505, -46.6333)
}

if location_method == "🏙️ Popular Cities":
    selected_city = st.sidebar.selectbox("Select your departure city:", list(POPULAR_CITIES.keys()), index=0)
    departure_coords = POPULAR_CITIES[selected_city]
    departure_location = selected_city
    
elif location_method == "📍 Enter Location":
    departure_location = st.sidebar.text_input(
        "Enter any location name:", 
        "New York City",
        help="Enter any city, landmark, or address - we'll find coordinates automatically!"
    )
    
    if st.sidebar.button("🔍 Find Coordinates"):
        with st.sidebar.container():
            with st.spinner("Finding coordinates..."):
                lat, lon, source = geocode_location_with_fallback(departure_location)
                
                if lat and lon:
                    departure_coords = (lat, lon)
                    st.sidebar.success(f"✅ Found coordinates using {source}")
                    st.sidebar.write(f"📍 {lat:.4f}, {lon:.4f}")
                else:
                    st.sidebar.error("❌ Could not find coordinates. Using default location.")
                    departure_coords = (40.7128, -74.0060)
    else:
        if departure_location:
            lat, lon, source = geocode_location_with_fallback(departure_location)
            if lat and lon:
                departure_coords = (lat, lon)
            else:
                departure_coords = (40.7128, -74.0060)
        else:
            departure_coords = (40.7128, -74.0060)
    
else:
    st.sidebar.write("Enter exact coordinates:")
    col_lat, col_lon = st.sidebar.columns(2)
    with col_lat:
        departure_lat = st.number_input("Latitude", value=40.7128, format="%.4f")
    with col_lon:
        departure_lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
    
    departure_coords = (departure_lat, departure_lon)
    departure_location = f"Custom ({departure_lat:.2f}, {departure_lon:.2f})"

st.sidebar.success(f"📍 Origin: {departure_location}")
st.sidebar.write(f"Coordinates: {departure_coords[0]:.4f}, {departure_coords[1]:.4f}")

transport_mode = st.sidebar.selectbox(
    "Preferred Transport",
    ["mixed", "car", "train", "plane"],
    help="How you'll travel between locations"
)

# ==================== AI MODEL FUNCTIONS ====================

def safe_extract_response_text(response):
    """Safely extract text from AI response"""
    if not response:
        return "No response received from AI model."
    
    try:
        if hasattr(response, 'content') and isinstance(response.content, str):
            return response.content
        elif hasattr(response, 'text') and isinstance(response.text, str):
            return response.text
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 1:
                    return "AI response was cut short. Try rephrasing your request."
                elif finish_reason == 2:
                    return "AI response exceeded token limit. Try breaking down your request."
                elif finish_reason == 3:
                    return "Response blocked for safety reasons. Please rephrase your request."
                elif finish_reason == 4:
                    return "Response blocked due to recitation concerns. Please try a different approach."
            
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts and len(parts) > 0:
                    text_parts = []
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    
                    if text_parts:
                        return '\n'.join(text_parts)
        
        if hasattr(response, 'content'):
            return str(response.content)
        elif hasattr(response, 'text'):
            return str(response.text)
        else:
            return str(response)
            
    except Exception as e:
        return f"Error extracting response: {str(e)}. The AI model may have returned an unexpected format."

def initialize_ai_model(provider: str, api_key: str, model_name: str):
    """Initialize AI model based on selected provider"""
    if not api_key:
        st.error(f"Please enter your {provider} API key!")
        return None
    
    try:
        if provider == "OpenAI":
            import openai
            os.environ["OPENAI_API_KEY"] = api_key
            from smolagents import OpenAIServerModel
            return OpenAIServerModel(model_name, max_tokens=3000)
            
        elif provider == "Google Gemini":
            import google.generativeai as genai
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            
            class GeminiModel:
                def __init__(self, model_name):
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                
                def __call__(self, messages, **kwargs):
                    if isinstance(messages, str):
                        prompt = messages
                    elif isinstance(messages, list) and len(messages) > 0:
                        if isinstance(messages[0], dict) and 'content' in messages[0]:
                            prompt = messages[0]['content']
                        else:
                            prompt = str(messages[0])
                    else:
                        prompt = str(messages)
                    
                    try:
                        response = self.model.generate_content(prompt)
                        
                        class SafeResponse:
                            def __init__(self, gemini_response):
                                self.original_response = gemini_response
                                self._content = None
                                self._text = None
                                self._extract_content()
                            
                            def _extract_content(self):
                                try:
                                    if (hasattr(self.original_response, 'candidates') and 
                                        self.original_response.candidates and
                                        hasattr(self.original_response.candidates[0], 'content') and
                                        hasattr(self.original_response.candidates[0].content, 'parts') and
                                        self.original_response.candidates[0].content.parts):
                                        
                                        parts = self.original_response.candidates[0].content.parts
                                        text_parts = [part.text for part in parts if hasattr(part, 'text') and part.text]
                                        
                                        if text_parts:
                                            self._content = '\n'.join(text_parts)
                                            self._text = self._content
                                        else:
                                            self._handle_no_content()
                                    else:
                                        self._handle_no_content()
                                except Exception as e:
                                    self._content = f"Error extracting content: {str(e)}"
                                    self._text = self._content
                            
                            def _handle_no_content(self):
                                finish_reason = None
                                try:
                                    if (hasattr(self.original_response, 'candidates') and 
                                        self.original_response.candidates):
                                        finish_reason = self.original_response.candidates[0].finish_reason
                                except:
                                    pass
                                
                                if finish_reason == 1:
                                    self._content = "AI stopped generating content. Please try rephrasing your request."
                                elif finish_reason == 2:
                                    self._content = "Response was too long. Please try a shorter request."
                                elif finish_reason == 3:
                                    self._content = "Response blocked for safety. Please rephrase your request."
                                elif finish_reason == 4:
                                    self._content = "Response blocked due to policy. Please try a different approach."
                                else:
                                    self._content = "No content generated. Please try rephrasing your request."
                                
                                self._text = self._content
                            
                            @property
                            def content(self):
                                return self._content
                            
                            @property
                            def text(self):
                                return self._text
                        
                        return SafeResponse(response)
                        
                    except Exception as e:
                        class ErrorResponse:
                            def __init__(self, error_msg):
                                self.content = f"Gemini model error: {error_msg}"
                                self.text = self.content
                        
                        return ErrorResponse(str(e))
            
            return GeminiModel(model_name)
            
        else:
            os.environ["TOGETHER_API_KEY"] = api_key
            from smolagents import InferenceClientModel
            return InferenceClientModel(model_name, provider="together", max_tokens=3000)
            
    except Exception as e:
        st.error(f"Error initializing {provider} model: {str(e)}")
        return None

def generate_ai_travel_plan(model, locations_df: pd.DataFrame, trip_days: int, daily_hours: int, budget: str):
    """Generate intelligent travel recommendations using AI"""
    if model is None:
        return "Please configure your AI model first."
    
    location_summary = locations_df.to_string(index=False)
    
    prompt = f"""
    As an expert travel planner, create an optimal {trip_days}-day itinerary for these destinations:
    
    LOCATIONS DATA:
    {location_summary}
    
    TRIP PARAMETERS:
    - Duration: {trip_days} days
    - Daily travel budget: {daily_hours} hours
    - Budget level: {budget}
    - Minimize travel time between locations
    - Group nearby locations together
    
    PROVIDE A COMPREHENSIVE PLAN INCLUDING:
    1. Day-by-day detailed itinerary with specific locations
    2. Travel time estimates and best routes between locations
    3. Recommended airports/transportation hubs
    4. Budget-appropriate accommodation suggestions
    5. Must-see vs optional attractions with priority rankings
    6. Local transportation tips
    7. Best times to visit each location
    8. Cultural tips and local customs to know
    9. Estimated daily costs for {budget} budget
    10. Emergency contacts and travel safety tips
    
    Format as a clear, actionable travel plan that's easy to follow.
    """
    
    try:
        response = None
        
        try:
            response = model(prompt)
        except (AttributeError, TypeError) as e:
            if "'str' object has no attribute 'role'" in str(e):
                messages = [{"role": "user", "content": prompt}]
                response = model(messages)
            else:
                raise e
        
        return safe_extract_response_text(response)
            
    except Exception as e:
        return f"Error generating AI plan: {str(e)}\n\nPlease check your AI model configuration."

def ai_enhanced_location_search(model, query: str, location_type: str):
    """Use AI to find and analyze travel destinations"""
    if model is None:
        return "Please configure your AI model first."
        
    prompt = f"""
    Find comprehensive information about travel destinations worldwide based on this query: "{query}"
    
    Focus on: {location_type} destinations
    
    For each destination provide:
    1. Exact name and key highlights
    2. Country and nearest major city/airport
    3. Main attractions and activities
    4. Best time to visit
    5. Approximate coordinates (latitude, longitude)
    6. Travel difficulty level and accessibility
    7. Budget range recommendations
    
    Provide at least 12-15 diverse destinations across different countries and continents.
    Include a mix of popular and hidden gem locations.
    
    Format as structured data that can be easily parsed.
    """
    
    try:
        response = None
        
        try:
            response = model(prompt)
        except (AttributeError, TypeError) as e:
            if "'str' object has no attribute 'role'" in str(e):
                messages = [{"role": "user", "content": prompt}]
                response = model(messages)
            else:
                raise e
        
        return safe_extract_response_text(response)
            
    except Exception as e:
        return f"Error in AI search: {str(e)}"

def calculate_travel_distance(coord1: Tuple[float, float], coord2: Tuple[float, float], transport_mode: str = "mixed") -> Dict:
    """Calculate distance and travel time between two coordinates"""
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    distance = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
    
    speeds = {"car": 70, "train": 100, "plane": 500, "mixed": 85}
    travel_time = distance / speeds.get(transport_mode, 85)
    
    return {
        "distance_km": round(distance, 2),
        "travel_time_hours": round(travel_time, 2),
        "transport_mode": transport_mode
    }

# ==================== MAIN APPLICATION ====================

# Show travel examples
st.header("🌟 Popular Travel Themes")
st.markdown("Get inspired by these curated travel themes or create your own custom adventure!")

# Display travel examples in a nice grid
example_cols = st.columns(3)
for i, (theme_name, theme_data) in enumerate(TRAVEL_EXAMPLES.items()):
    with example_cols[i % 3]:
        with st.container():
            st.markdown(f"""
            <div class="example-card">
                <h4>{theme_name}</h4>
                <p>{theme_data['description']}</p>
                <small>{len(theme_data['locations'])} destinations</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Use {theme_name}", key=f"example_{i}"):
                # Load example locations into session state
                location_data = []
                for name, (lat, lon) in theme_data['locations'].items():
                    travel_info = calculate_travel_distance(departure_coords, (lat, lon), transport_mode)
                    
                    # Extract country from location name
                    country = name.split(', ')[-1] if ', ' in name else "Unknown"
                    
                    location_data.append({
                        "Location": name,
                        "Country": country,
                        "Type": theme_name.split(' ')[-1],  # Extract theme type
                        "Latitude": lat,
                        "Longitude": lon,
                        "Distance (km)": travel_info["distance_km"],
                        "Travel Time (hrs)": travel_info["travel_time_hours"],
                        "Transport": travel_info["transport_mode"]
                    })
                
                df = pd.DataFrame(location_data)
                df = clean_dataframe_columns(df)
                
                st.session_state.locations_df = df
                st.session_state.departure_info = {
                    "location": departure_location,
                    "coords": departure_coords
                }
                
                st.success(f"✅ Loaded {len(location_data)} destinations from {theme_name}!")
                st.rerun()

# Main application layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🗺️ Travel Destinations")
    
    # AI Model initialization
    if 'ai_key' in locals() and ai_key:
        model = initialize_ai_model(ai_provider, ai_key, model_name)
        if model:
            st.success(f"✅ {ai_provider} {model_name} initialized successfully!")
    else:
        model = None
        st.warning(f"⚠️ Please enter your {ai_provider} API key to enable AI features")
    
    # Enhanced search options
    search_option = st.radio(
        "How would you like to plan your trip?",
        ["🎯 Use Travel Themes Above", "🤖 AI-Powered Destination Search", "🌐 Web Search + AI Analysis", "📍 Custom Destinations + Discovery"],
        help="Choose your preferred method to find amazing destinations"
    )

    # CUSTOM DESTINATIONS WITH NEARBY DISCOVERY
    if search_option == "📍 Custom Destinations + Discovery":
        st.subheader("📍 Create Your Custom Travel Experience")
        st.info("💡 **New Feature**: Add any destination and we'll discover amazing nearby places to visit!")
        
        # Initialize custom locations in session state
        if 'custom_locations_input' not in st.session_state:
            st.session_state.custom_locations_input = []
        
        # Form to add custom locations
        with st.form("add_custom_destination"):
            col1_form, col2_form = st.columns(2)
            with col1_form:
                custom_name = st.text_input("Destination Name", placeholder="e.g., Tokyo, Paris, Grand Canyon, Machu Picchu")
                custom_country = st.text_input("Country (optional)", placeholder="e.g., Japan, France, USA, Peru")
            with col2_form:
                discovery_method = st.selectbox(
                    "Nearby Places Discovery",
                    ["AI + Web APIs", "AI Only", "Web APIs Only", "None"],
                    help="How to find nearby attractions"
                )
                radius_km = st.slider("Search Radius (km)", 10, 200, 50, help="How far to search for nearby places")
            
            custom_description = st.text_area("What interests you? (optional)", placeholder="Historical sites, nature, food, adventure, culture...")
            
            submitted = st.form_submit_button("➕ Add Destination & Discover Nearby Places")
            
            if submitted and custom_name:
                with st.spinner(f"Finding {custom_name} and discovering nearby attractions..."):
                    # Auto-geocode the location
                    lat, lon, source = geocode_location_with_fallback(custom_name)
                    
                    if lat and lon:
                        # Clear previous locations for new search
                        st.session_state.custom_locations_input = []
                        
                        # Add main destination
                        main_location = {
                            "name": custom_name,
                            "country": custom_country or custom_name.split(', ')[-1] if ', ' in custom_name else "Unknown",
                            "lat": lat,
                            "lon": lon,
                            "description": custom_description,
                            "geocoded_source": source,
                            "type": "Main Destination"
                        }
                        st.session_state.custom_locations_input.append(main_location)
                        
                        # Find nearby places
                        nearby_places = []
                        
                        if discovery_method in ["AI + Web APIs", "AI Only"]:
                            if model:
                                with st.spinner("AI is discovering nearby attractions..."):
                                    ai_places = find_nearby_places_with_ai(model, custom_name, lat, lon, radius_km)
                                    nearby_places.extend(ai_places)
                                    if ai_places:
                                        st.success(f"🤖 AI found {len(ai_places)} nearby places!")
                            else:
                                st.warning("AI model not configured. Using web APIs only.")
                        
                        if discovery_method in ["AI + Web APIs", "Web APIs Only"]:
                            with st.spinner("Searching databases for nearby attractions..."):
                                web_places = find_nearby_places_with_api(lat, lon, radius_km)
                                nearby_places.extend(web_places)
                                if web_places:
                                    st.success(f"🌐 Found {len(web_places)} places from web databases!")
                        
                        # Remove duplicates and add nearby places
                        seen_names = {main_location["name"].lower()}
                        unique_nearby = []
                        
                        for place in nearby_places:
                            place_name_lower = place["name"].lower()
                            if place_name_lower not in seen_names and len(place_name_lower) > 2:
                                seen_names.add(place_name_lower)
                                place["country"] = main_location["country"]
                                place["type"] = f"Nearby {place.get('type', 'Attraction')}"
                                unique_nearby.append(place)
                        
                        # Add up to 15 nearby places
                        st.session_state.custom_locations_input.extend(unique_nearby[:15])
                        
                        total_found = len(st.session_state.custom_locations_input)
                        st.success(f"✅ Found {custom_name} + {total_found-1} amazing nearby places!")
                        st.rerun()
                    else:
                        st.error(f"❌ Could not find coordinates for '{custom_name}'. Please try a more specific location name.")
        
        # Display current custom locations
        if st.session_state.custom_locations_input:
            st.write(f"**Your Travel Destinations ({len(st.session_state.custom_locations_input)} total):**")
            
            # Separate main and nearby locations
            main_locations = [loc for loc in st.session_state.custom_locations_input if loc.get("type") == "Main Destination"]
            nearby_locations = [loc for loc in st.session_state.custom_locations_input if loc.get("type") != "Main Destination"]
            
            # Display main destination
            for i, loc in enumerate(main_locations):
                st.markdown(f"**🎯 Main Destination:**")
                col_info, col_remove = st.columns([5, 1])
                with col_info:
                    st.write(f"• **{loc['name']}** ({loc['country']}) - {loc['lat']:.4f}, {loc['lon']:.4f}")
                    if loc['description']:
                        st.write(f"  _{loc['description']}_")
                    st.caption(f"📍 Coordinates found via: {loc.get('geocoded_source', 'manual')}")
                with col_remove:
                    if st.button("🗑️ Clear All", key=f"remove_main_{i}"):
                        st.session_state.custom_locations_input = []
                        st.rerun()
            
            # Display nearby places
            if nearby_locations:
                st.markdown(f"**🌟 Nearby Places to Visit ({len(nearby_locations)}):**")
                
                # Show in a more compact format
                for i in range(0, len(nearby_locations), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(nearby_locations):
                            loc = nearby_locations[i + j]
                            with col:
                                st.write(f"**{loc['name']}**")
                                st.caption(f"{loc.get('type', 'Attraction')} | {loc.get('source', 'unknown')}")
                                if st.button("❌", key=f"remove_nearby_{i+j}", help="Remove this place"):
                                    # Find index in main list and remove
                                    for idx, main_loc in enumerate(st.session_state.custom_locations_input):
                                        if main_loc['name'] == loc['name']:
                                            st.session_state.custom_locations_input.pop(idx)
                                            break
                                    st.rerun()
            
            # Button to process current locations
            if st.button("🚀 Create Travel Plan with These Destinations", key="process_custom"):
                with st.spinner("Processing your custom destinations..."):
                    all_locations = {}
                    for loc in st.session_state.custom_locations_input:
                        all_locations[loc["name"]] = (loc["lat"], loc["lon"], loc["country"])
                    
                    # Process locations
                    location_data = []
                    for name, (lat, lon, country) in all_locations.items():
                        travel_info = calculate_travel_distance(departure_coords, (lat, lon), transport_mode)
                        
                        # Find location type
                        location_type = "Nearby Attraction"
                        for loc in st.session_state.custom_locations_input:
                            if loc["name"] == name:
                                location_type = loc.get("type", "Attraction")
                                break
                        
                        location_data.append({
                            "Location": name,
                            "Country": country,
                            "Type": location_type,
                            "Latitude": lat,
                            "Longitude": lon,
                            "Distance (km)": travel_info["distance_km"],
                            "Travel Time (hrs)": travel_info["travel_time_hours"],
                            "Transport": travel_info["transport_mode"]
                        })
                    
                    df = pd.DataFrame(location_data)
                    df = clean_dataframe_columns(df)
                    
                    st.session_state.locations_df = df
                    st.session_state.departure_info = {
                        "location": departure_location,
                        "coords": departure_coords
                    }
                    
                    main_count = len([loc for loc in location_data if "Main Destination" in loc["Type"]])
                    nearby_count = len(location_data) - main_count
                    st.success(f"✅ Created travel plan with {len(all_locations)} destinations! ({main_count} main, {nearby_count} nearby)")
                    st.rerun()

    # AI-powered search options
    elif search_option == "🤖 AI-Powered Destination Search":
        st.subheader("🤖 AI Travel Discovery")
        
        col_search1, col_search2 = st.columns(2)
        with col_search1:
            search_query = st.text_input(
                "What kind of trip do you want?", 
                "Amazing cultural destinations in Asia",
                help="Describe your ideal trip - be specific!"
            )
            
            trip_type = st.selectbox(
                "Trip Style",
                ["Cultural & Historical", "Adventure & Nature", "Beach & Relaxation", "City & Urban", "Food & Culinary", "Art & Architecture", "Spiritual & Wellness", "Off-the-beaten-path"]
            )
        
        with col_search2:
            continent_filter = st.multiselect(
                "Preferred Continents (optional)",
                ["Asia", "Europe", "North America", "South America", "Africa", "Oceania", "Antarctica"],
                help="Leave empty for worldwide search"
            )
            
            difficulty_level = st.selectbox(
                "Travel Difficulty",
                ["Easy (tourist-friendly)", "Moderate (some planning needed)", "Challenging (adventure required)"]
            )
        
        if st.button("🔍 Discover Amazing Destinations", key="ai_search_btn"):
            if model:
                with st.spinner("AI is finding perfect destinations for you..."):
                    # Enhanced search query
                    enhanced_query = f"{search_query} - Focus on {trip_type} destinations"
                    if continent_filter:
                        enhanced_query += f" in {', '.join(continent_filter)}"
                    enhanced_query += f" with {difficulty_level} accessibility"
                    
                    ai_results = ai_enhanced_location_search(model, enhanced_query, trip_type)
                    st.text_area("🤖 AI Discovery Results:", ai_results, height=300)
                    
                    # Try to parse and create locations (basic implementation)
                    st.info("💡 AI has found amazing destinations! Copy interesting location names to the Custom Destinations section above to add them to your trip.")
            else:
                st.error("Please configure your AI model first!")
    
    elif search_option == "🌐 Web Search + AI Analysis":
        st.subheader("🌐 Advanced Web Search + AI Analysis")
        st.info("This feature combines real-time web search with AI analysis for the most up-to-date travel information.")
        
        search_query = st.text_input(
            "Search for travel destinations:", 
            "Best travel destinations 2025",
            help="Enter your travel search query"
        )
        
        if st.button("🔍 Search Web + Analyze with AI", key="web_search_btn"):
            if model:
                st.warning("🚧 Web search integration requires additional API setup. For now, please use the AI-powered search or custom destinations options.")
            else:
                st.error("Please configure your AI model first!")
    
    else:  # Use Travel Themes Above
        st.info("👆 Click on any travel theme above to instantly load those destinations, or use the other search options below.")

    # Display results if available
    if 'locations_df' in st.session_state:
        st.subheader("📊 Your Travel Destinations")
        
        # Show departure info
        if 'departure_info' in st.session_state:
            dep_info = st.session_state.departure_info
            st.info(f"📍 Origin: {dep_info['location']} ({dep_info['coords'][0]:.2f}, {dep_info['coords'][1]:.2f})")
        
        # Display dataframe with better formatting
        df_display = st.session_state.locations_df.copy()
        df_display = df_display.round(2)
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Download options
        col_download1, col_download2 = st.columns(2)
        with col_download1:
            csv = st.session_state.locations_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"travel_destinations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col_download2:
            # Create a simple text itinerary
            text_itinerary = f"Travel Itinerary - {datetime.now().strftime('%Y-%m-%d')}\n"
            text_itinerary += f"Origin: {st.session_state.departure_info['location']}\n"
            text_itinerary += f"Trip Duration: {trip_days} days\n"
            text_itinerary += f"Budget Level: {budget_range}\n\n"
            
            for _, row in st.session_state.locations_df.iterrows():
                text_itinerary += f"• {row['location']} ({row['country']})\n"
                text_itinerary += f"  Distance: {row['distance (km)']} km, Travel time: {row['travel time (hrs)']} hrs\n"
                text_itinerary += f"  Type: {row['type']}\n\n"
            
            st.download_button(
                label="📝 Download Itinerary",
                data=text_itinerary,
                file_name=f"travel_itinerary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

with col2:
    st.header("📈 Trip Analytics")
    
    if 'locations_df' in st.session_state:
        df = st.session_state.locations_df
        
        if any(col != col.lower().strip() for col in df.columns):
            df = clean_dataframe_columns(df)
            st.session_state.locations_df = df
        
        # Enhanced statistics with styling
        total_locations = len(df)
        avg_distance = df['distance (km)'].mean()
        closest_location = df.loc[df['distance (km)'].idxmin(), 'location']
        total_travel_time = df['travel time (hrs)'].sum()
        
        # Styled metrics
        st.markdown(f"""
        <div class="stat-card">
            <h3>{total_locations}</h3>
            <p>Total Destinations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <h3>{avg_distance:.0f} km</h3>
            <p>Average Distance</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <h3>{total_travel_time:.1f} hrs</h3>
            <p>Total Travel Time</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**🎯 Closest Destination:** {closest_location}")
        
        # Location type breakdown
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            st.write("**📊 Destination Types:**")
            for loc_type, count in type_counts.items():
                percentage = (count / total_locations) * 100
                st.write(f"• {loc_type}: {count} ({percentage:.0f}%)")
        
        # Enhanced feasibility analysis
        max_daily_distance = daily_budget_hours * 70
        feasible_locations = df[df['distance (km)'] <= max_daily_distance * trip_days]
        
        st.subheader("🎯 Trip Feasibility")
        st.write(f"**Feasible for {trip_days}-day trip:** {len(feasible_locations)}/{total_locations}")
        st.write(f"**Daily travel budget:** {daily_budget_hours} hours")
        st.write(f"**Max daily distance:** {max_daily_distance:.0f} km")
        
        feasibility_score = len(feasible_locations) / len(df) if len(df) > 0 else 0
        st.progress(feasibility_score)
        
        if feasibility_score >= 0.8:
            st.success(f"✅ Excellent feasibility ({feasibility_score:.0%})")
        elif feasibility_score >= 0.6:
            st.warning(f"⚠️ Good feasibility ({feasibility_score:.0%})")
        else:
            st.error(f"❌ Challenging feasibility ({feasibility_score:.0%})")
        
        # Budget estimation
        st.subheader("💰 Budget Estimation")
        budget_multipliers = {
            "Budget ($)": 50,
            "Mid-range ($)": 100,
            "Luxury ($$)": 200,
            "Ultra-luxury ($$)": 500
        }
        
        daily_budget = budget_multipliers.get(budget_range, 100)
        total_budget = daily_budget * trip_days
        
        st.write(f"**Estimated daily cost:** ${daily_budget}")
        st.write(f"**Total trip budget:** ${total_budget:,}")
        st.caption("*Estimates include accommodation, food, and activities")

# Enhanced map visualization
if 'locations_df' in st.session_state:
    st.header("🗺️ Interactive Travel Map")
    
    df = st.session_state.locations_df
    
    if any(col != col.lower().strip() for col in df.columns):
        df = clean_dataframe_columns(df)
    
    if df.empty:
        st.error("No location data available. Please search for destinations first.")
    else:
        st.write(f"📍 Showing {len(df)} destinations on interactive map")
        
        try:
            # Enhanced color mapping for different destination types
            color_discrete_map = {
                "main destination": "#FF6B35",
                "nearby attraction": "#2E8B57",
                "nearby cultural": "#9932CC",
                "nearby historical": "#4169E1",
                "nearby natural": "#228B22",
                "nearby entertainment": "#FF1493",
                "cultural & historical": "#8B4513",
                "adventure & nature": "#006400",
                "beach & relaxation": "#00CED1",
                "city & urban": "#4682B4",
                "food & culinary": "#FF8C00",
                "art & architecture": "#9370DB",
            }
            
            # Create enhanced map
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                hover_name="location",
                hover_data=["country", "type", "distance (km)", "travel time (hrs)"],
                color="type" if 'type' in df.columns else "country",
                size="distance (km)",
                color_discrete_map=color_discrete_map,
                size_max=30,
                zoom=1,
                height=700,
                title="🌍 Your Travel Destinations Map",
                labels={
                    "distance (km)": "Distance from Origin (km)",
                    "travel time (hrs)": "Travel Time (hours)",
                    "type": "Destination Type"
                }
            )
            
            # Add origin marker with enhanced styling
            if 'departure_info' in st.session_state:
                dep_coords = st.session_state.departure_info['coords']
                dep_location = st.session_state.departure_info['location']
                
                fig.add_trace(go.Scattermapbox(
                    lat=[dep_coords[0]],
                    lon=[dep_coords[1]],
                    mode='markers',
                    marker=dict(
                        size=25, 
                        color='red', 
                        symbol='star',
                        opacity=0.9
                    ),
                    text=[f"🏠 Origin: {dep_location}"],
                    hoverinfo='text',
                    name="🏠 Your Origin",
                    showlegend=True
                ))
            
            # Enhanced layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    bearing=0,
                    pitch=0,
                    zoom=1,
                    center=dict(
                        lat=df['latitude'].mean(),
                        lon=df['longitude'].mean()
                    )
                ),
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Map controls
            col_map1, col_map2, col_map3 = st.columns(3)
            with col_map1:
                if st.button("🌍 Zoom to Fit All"):
                    st.rerun()
            with col_map2:
                if st.button("📍 Center on Origin"):
                    st.rerun()
            with col_map3:
                map_style = st.selectbox("Map Style", ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain"])
            
        except Exception as e:
            st.error(f"Error displaying interactive map: {e}")
            # Fallback simple map
            st.subheader("📍 Simple Map View")
            map_data = df[['latitude', 'longitude', 'location']].copy()
            map_data = map_data.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            st.map(map_data, zoom=2)

# AI-powered itinerary planner
if 'locations_df' in st.session_state:
    st.header("🤖 AI Travel Planner")
    
    col_ai1, col_ai2 = st.columns([2, 1])
    
    with col_ai1:
        if 'plan_counter' not in st.session_state:
            st.session_state.plan_counter = 0
            
        generate_plan = st.button("✨ Generate AI Travel Plan", key=f"ai_plan_{st.session_state.plan_counter}")
        
        if generate_plan:
            if model:
                with st.spinner("🤖 AI is crafting your perfect itinerary..."):
                    ai_plan = generate_ai_travel_plan(
                        model, 
                        st.session_state.locations_df, 
                        trip_days, 
                        daily_budget_hours,
                        budget_range
                    )
                    st.session_state.travel_plan = ai_plan
                    st.session_state.plan_counter += 1
                    st.rerun()
            else:
                st.error("Please configure your AI model first!")
    
    with col_ai2:
        ai_model_info = model_name if 'model_name' in locals() else 'None'
        st.info(f"🎯 Using {ai_provider} {ai_model_info}")
        st.write(f"📅 {trip_days} days, {daily_budget_hours}h/day")
        st.write(f"💰 {budget_range} budget")
    
    # Display AI-generated plan
    if 'travel_plan' in st.session_state:
        st.subheader("🗓️ Your Personalized AI Itinerary")
        
        # Show the plan in an expandable section
        with st.expander("📋 Full Itinerary (Click to expand)", expanded=True):
            st.markdown(st.session_state.travel_plan)
        
        # Plan management buttons
        col_plan1, col_plan2, col_plan3 = st.columns(3)
        
        with col_plan1:
            regenerate_plan = st.button("🔄 Regenerate Plan", key=f"regen_plan_{st.session_state.plan_counter}")
            
            if regenerate_plan:
                if model:
                    with st.spinner("Creating alternative plan..."):
                        ai_plan = generate_ai_travel_plan(
                            model, 
                            st.session_state.locations_df, 
                            trip_days, 
                            daily_budget_hours,
                            budget_range
                        )
                        st.session_state.travel_plan = ai_plan
                        st.session_state.plan_counter += 1
                        st.rerun()
                else:
                    st.error("Please configure your AI model first!")
        
        with col_plan2:
            # Download the AI plan
            plan_text = f"AI Travel Plan - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            plan_text += st.session_state.travel_plan
            
            st.download_button(
                label="💾 Download Plan",
                data=plan_text,
                file_name=f"ai_travel_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        
        with col_plan3:
            if st.button("🗑️ Clear Plan"):
                if 'travel_plan' in st.session_state:
                    del st.session_state.travel_plan
                st.rerun()

# Traditional itinerary planner (fallback)
if 'locations_df' in st.session_state:
    st.header("📅 Basic Itinerary Planner")
    
    df = st.session_state.locations_df
    
    if any(col != col.lower().strip() for col in df.columns):
        df = clean_dataframe_columns(df)
    
    df_sorted = df.sort_values('distance (km)')
    
    # Group locations by country and proximity
    countries = df_sorted['country'].unique()
    
    st.write(f"**Suggested {trip_days}-day itinerary based on proximity:**")
    
    for i, country in enumerate(countries[:trip_days]):
        with st.expander(f"Day {i+1}: Explore {country}", expanded=i<3):
            country_locations = df_sorted[df_sorted['country'] == country]
            
            st.write(f"**🎯 Recommended destinations in {country}:**")
            for j, (_, location) in enumerate(country_locations.head(4).iterrows()):
                location_type = f" ({location['type']})" if 'type' in location else ""
                priority = "🔥 Must-see" if j == 0 else "⭐ Worth visiting"
                
                st.write(f"• **{location['location']}**{location_type}")
                st.write(f"  {priority} - {location['distance (km)']} km from origin")
            
            total_time = country_locations.head(4)['travel time (hrs)'].sum()
            st.write(f"**⏱️ Estimated travel time:** {total_time:.1f} hours")
            
            if total_time > daily_budget_hours:
                st.warning(f"⚠️ This exceeds your daily travel budget of {daily_budget_hours} hours")
            else:
                st.success(f"✅ Within your daily travel budget")

# Travel tips and recommendations
if 'locations_df' in st.session_state:
    st.header("💡 Smart Travel Tips")
    
    df = st.session_state.locations_df
    
    # Analyze the destinations and provide tips
    col_tip1, col_tip2 = st.columns(2)
    
    with col_tip1:
        st.subheader("🎒 Packing Tips")
        
        # Determine climate zones
        climates = []
        for _, row in df.iterrows():
            lat = row['latitude']
            if lat > 60 or lat < -60:
                climates.append("polar")
            elif 23.5 <= lat <= 60 or -60 <= lat <= -23.5:
                climates.append("temperate")
            else:
                climates.append("tropical")
        
        unique_climates = list(set(climates))
        
        if "polar" in unique_climates:
            st.write("❄️ **Cold Weather Gear:** Heavy coat, warm layers, waterproof boots")
        if "temperate" in unique_climates:
            st.write("🌤️ **Versatile Clothing:** Layers, light jacket, comfortable walking shoes")
        if "tropical" in unique_climates:
            st.write("☀️ **Hot Weather Essentials:** Light clothing, sunscreen, hat, sandals")
        
        st.write("📱 **Tech Essentials:** Universal adapter, portable charger, offline maps")
        st.write("💊 **Health Kit:** Basic medications, hand sanitizer, travel insurance")
    
    with col_tip2:
        st.subheader("🚗 Transportation Tips")
        
        max_distance = df['distance (km)'].max()
        
        if max_distance > 5000:
            st.write("✈️ **International Travel:** Consider flights for long distances")
            st.write("📋 **Documentation:** Check visa requirements and passport validity")
        elif max_distance > 1000:
            st.write("🚂 **Regional Travel:** Mix of flights and ground transport")
            st.write("🎫 **Booking:** Book transportation in advance for better prices")
        else:
            st.write("🚗 **Ground Transport:** Perfect for road trips or train journeys")
            st.write("⛽ **Planning:** Plan fuel stops and rest breaks")
        
        st.write("📱 **Apps:** Download transport apps, translation apps, currency converters")
        st.write("💳 **Payment:** Notify bank of travel, carry multiple payment methods")

# Footer with enhanced information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h3>🌍 Universal AI Travel Planner</h3>
    <p><strong>Features:</strong></p>
    <p>✅ Auto-Geocoding | 🤖 AI-Powered Discovery | 🗺️ Interactive Maps | 📊 Smart Analytics</p>
    <p>✅ Nearby Places Detection | 💰 Budget Planning | 📱 Mobile-Friendly | 🌐 Worldwide Coverage</p>
    <br>
    <p><em>"Travel is the only thing you buy that makes you richer."</em></p>
    <p>🌟 <strong>Happy Travels!</strong> 🌟</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Deployment instructions in sidebar
with st.sidebar.expander("🚀 Deployment & Setup Guide"):
    st.markdown("""
    ## 🚀 **Quick Deploy Options**
    
    ### **1. Streamlit Cloud (Recommended)**
    ```bash
    # 1. Push to GitHub
    # 2. Go to streamlit.io
    # 3. Connect repository
    # 4. Add API keys in secrets
    # 5. Deploy in 1-click!
    ```
    
    ### **2. Local Development**
    ```bash
    pip install streamlit plotly pandas requests
    streamlit run streamlit_app.py
    ```
    
    ### **3. Docker Deployment**
    ```bash
    docker build -t travel-planner .
    docker run -p 8501:8501 travel-planner
    ```
    
    ---
    
    ## 🔐 **API Keys Required**
    
    **AI Models (Choose One):**
    - 🤖 **OpenAI**: platform.openai.com
    - 🧠 **Google Gemini**: ai.google.dev (FREE tier!)
    - 🔓 **Together AI**: together.ai (Open source models)
    
    **Free Features (No API needed):**
    - 🗺️ Auto-Geocoding (Open-Meteo + OpenStreetMap)
    - 🔍 Nearby Places Discovery (Overpass API)
    - 📊 Analytics & Mapping (Plotly)
    - 📱 Interactive Maps
    
    ---
    
    ## 🆕 **New Universal Features**
    
    ✅ **Any Destination Worldwide**
    ✅ **6 Curated Travel Themes**
    ✅ **AI-Powered Destination Discovery**
    ✅ **Smart Budget Planning**
    ✅ **Climate-Based Packing Tips**
    ✅ **Feasibility Analysis**
    ✅ **Enhanced Interactive Maps**
    ✅ **Multi-Format Downloads**
    
    ---
    
    ## 💡 **Pro Tips**
    
    - **Free Setup**: Use Gemini Free + Auto-Geocoding
    - **Best Performance**: OpenAI GPT-4o + All APIs
    - **Open Source**: Together AI + Free Services
    - **Mobile**: Works great on phones/tablets
    - **Offline**: Download your itinerary as PDF/text
    """)

# Requirements.txt generator
with st.sidebar.expander("📦 Requirements.txt"):
    requirements = """streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
requests>=2.31.0
google-generativeai>=0.3.0
openai>=1.0.0
smolagents>=0.1.0"""
    
    st.code(requirements, language="text")
    
    st.download_button(
        label="📥 Download requirements.txt",
        data=requirements,
        file_name="requirements.txt",
        mime="text/plain"
    )

# App usage statistics (if you want to track usage)
if 'usage_stats' not in st.session_state:
    st.session_state.usage_stats = {
        'sessions': 0,
        'destinations_added': 0,
        'ai_plans_generated': 0
    }

# Increment session counter
if 'session_counted' not in st.session_state:
    st.session_state.usage_stats['sessions'] += 1
    st.session_state.session_counted = True

# Show usage stats in sidebar (optional)
with st.sidebar.expander("📈 Usage Statistics"):
    stats = st.session_state.usage_stats
    st.metric("Sessions", stats['sessions'])
    st.metric("Destinations Added", stats.get('destinations_added', 0))
    st.metric("AI Plans Generated", stats.get('ai_plans_generated', 0))

if __name__ == "__main__":
    # This runs when script is executed directly
    st.balloons()  # Welcome celebration for new users!
