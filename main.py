# streamlit_app.py
"""
Harry Potter Travel Planner - Enhanced Streamlit Web App with Auto-Geocoding & Nearby Places Discovery
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
    page_title="AI Travel Planner (Harry Potter filming location default)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #722F37;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #8B4513;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚡ AI Travel Planner (Harry Potter filming location default) ⚡</h1>', unsafe_allow_html=True)
st.markdown("Plan your magical journey to filming locations worldwide! Now with **automatic location geocoding** and **nearby places discovery** - just enter location names!")

# ==================== HELPER FUNCTIONS ====================

def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame columns by removing spaces and converting to lowercase"""
    df.columns = df.columns.str.strip().str.lower()
    return df

def safe_get_column(df, column_name, default_name=None):
    """Safely get column from DataFrame with case-insensitive matching"""
    # Create mapping of lowercase column names to actual names
    col_map = {col.lower().strip(): col for col in df.columns}
    
    # Try to find the column (case-insensitive)
    target_col = column_name.lower().strip()
    if target_col in col_map:
        return col_map[target_col]
    elif default_name and default_name in df.columns:
        return default_name
    else:
        raise KeyError(f"Column '{column_name}' not found. Available columns: {list(df.columns)}")

# ==================== GEOCODING FUNCTIONS ====================

def geocode_location_with_fallback(location: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Try multiple free geocoding APIs as fallback
    Returns: (lat, lon, source_api)
    """
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
        params = {
            "q": location,
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "Harry Potter Travel Planner"}
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
    - Find 10 diverse attractions, landmarks, or points of interest
    - Include places within approximately {radius_km}km radius
    - For each place provide: exact name, approximate coordinates (lat, lon), type of attraction
    - Include mix of: historical sites, natural attractions, cultural sites, entertainment venues
    - Avoid duplicating the main location: {location_name}
    
    FORMAT RESPONSE AS:
    1. Place Name | Latitude | Longitude | Type | Distance from main location
    2. Place Name | Latitude | Longitude | Type | Distance from main location
    ...
    
    Example:
    1. Edinburgh Castle | 55.9486 | -3.1999 | Historical | 2km
    2. Royal Mile | 55.9500 | -3.1900 | Cultural | 1km
    
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
                    # Extract name (remove numbering)
                    name = parts[0]
                    # Remove leading numbers and dots
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
    
    return places[:10]  # Limit to 10 places

def find_nearby_places_with_api(lat: float, lon: float, radius_km: int = 50):
    """Find nearby places using free web APIs"""
    places = []
    
    try:
        # Use Overpass API (OpenStreetMap) to find nearby attractions
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Query for tourist attractions, museums, monuments, etc.
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["tourism"~"attraction|museum|monument|castle|gallery"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
          node["historic"~"castle|monument|museum|ruins"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
          node["amenity"~"theatre|cinema|library"]({lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5});
        );
        out center meta;
        """
        
        response = requests.post(overpass_url, data=overpass_query, timeout=30)
        data = response.json()
        
        if 'elements' in data:
            for element in data['elements'][:15]:  # Limit results
                if 'tags' in element and 'name' in element['tags']:
                    place_lat = element.get('lat', 0)
                    place_lon = element.get('lon', 0)
                    name = element['tags']['name']
                    
                    # Determine type
                    tags = element['tags']
                    place_type = tags.get('tourism', tags.get('historic', tags.get('amenity', 'attraction')))
                    
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
    
    return places[:10]

# ==================== SIDEBAR CONFIGURATION ====================

st.sidebar.header("🧳 Trip Configuration")

# API Keys setup
with st.sidebar.expander("🔐 API Keys Setup"):
    st.info("Enter your API keys to enable full functionality")
    
    # AI Model Provider Selection
    ai_provider = st.selectbox(
        "🤖 Choose AI Provider",
        ["OpenAI", "Google Gemini", "Together AI"],
        help="Select your preferred AI model provider"
    )
    
    # Conditional API key inputs based on provider
    if ai_provider == "OpenAI":
        ai_key = st.text_input("OpenAI API Key", type="password", help="For GPT models")
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    elif ai_provider == "Google Gemini":
        ai_key = st.text_input("Google AI API Key", type="password", help="For Gemini models")
        model_name = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"])
    else:  # Together AI
        ai_key = st.text_input("Together AI API Key", type="password", help="For open-source models")
        model_name = st.selectbox("Model", ["Qwen/Qwen2.5-Coder-32B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"])
    
    # Search API keys
    st.markdown("**Search APIs:**")
    serpapi_key = st.text_input("SerpAPI Key", type="password", help="For web search")
    serper_key = st.text_input("Serper API Key", type="password", help="Alternative search API")

# Trip parameters
trip_days = st.sidebar.slider("Trip Duration (days)", 2, 7, 4)
daily_budget_hours = st.sidebar.slider("Daily Travel Budget (hours)", 2, 8, 4)

# Enhanced origin location selection with automatic geocoding
st.sidebar.subheader("🌍 Origin Location")
location_method = st.sidebar.radio(
    "Select Origin Method:",
    ["🏙️ Choose from Popular Cities", "📍 Enter Location Name", "🗺️ Use Exact Coordinates"]
)

# Popular cities with coordinates
POPULAR_CITIES = {
    "London, UK": (51.5074, -0.1278),
    "New York, USA": (40.7128, -74.0060),
    "Paris, France": (48.8566, 2.3522),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Mumbai, India": (19.0760, 72.8777),
    "Delhi, India": (28.6139, 77.2090),
    "Kolkata, India": (22.5744, 88.3629),
    "Bangalore, India": (12.9716, 77.5946),
    "Dubai, UAE": (25.2048, 55.2708),
    "Singapore": (1.3521, 103.8198),
    "Berlin, Germany": (52.5200, 13.4050),
    "Rome, Italy": (41.9028, 12.4964),
    "Madrid, Spain": (40.4168, -3.7038),
    "Amsterdam, Netherlands": (52.3676, 4.9041)
}

if location_method == "🏙️ Choose from Popular Cities":
    selected_city = st.sidebar.selectbox(
        "Select your departure city:",
        list(POPULAR_CITIES.keys()),
        index=7  # Default to Kolkata
    )
    departure_coords = POPULAR_CITIES[selected_city]
    departure_location = selected_city
    
elif location_method == "📍 Enter Location Name":
    # Automatic geocoding for user input
    departure_location = st.sidebar.text_input(
        "Enter any location name:", 
        "Edinburgh, Scotland",
        help="Enter any city, landmark, or address - we'll find the coordinates automatically!"
    )
    
    # Geocode button for manual trigger
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
                    departure_coords = (22.5744, 88.3629)  # Default to Kolkata
    else:
        # Auto-geocode on page load if location is set
        if departure_location and departure_location != "":
            lat, lon, source = geocode_location_with_fallback(departure_location)
            if lat and lon:
                departure_coords = (lat, lon)
            else:
                departure_coords = (22.5744, 88.3629)  # Default
        else:
            departure_coords = (22.5744, 88.3629)
    
else:  # Use Exact Coordinates
    st.sidebar.write("Enter exact coordinates:")
    col_lat, col_lon = st.sidebar.columns(2)
    with col_lat:
        departure_lat = st.number_input("Latitude", value=22.5744, format="%.4f")
    with col_lon:
        departure_lon = st.number_input("Longitude", value=88.3629, format="%.4f")
    
    departure_coords = (departure_lat, departure_lon)
    departure_location = f"Custom ({departure_lat:.2f}, {departure_lon:.2f})"

# Display selected origin
st.sidebar.success(f"📍 Origin: {departure_location}")
st.sidebar.write(f"Coordinates: {departure_coords[0]:.4f}, {departure_coords[1]:.4f}")

# Transport preferences
transport_mode = st.sidebar.selectbox(
    "Preferred Ground Transport",
    ["car", "train", "mixed"],
    help="How you'll travel between locations"
)

# ==================== AI MODEL FUNCTIONS ====================

def safe_extract_response_text(response):
    """Safely extract text from Gemini response, handling finish_reason and empty parts"""
    
    if not response:
        return "No response received from AI model."
    
    try:
        # Check if it's a direct text response
        if hasattr(response, 'content') and isinstance(response.content, str):
            return response.content
        elif hasattr(response, 'text') and isinstance(response.text, str):
            return response.text
        
        # For Gemini API responses, check candidates and finish_reason
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check finish_reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 1:
                    return "AI response was cut short (natural stop). Try rephrasing your request or reducing the complexity."
                elif finish_reason == 2:
                    return "AI response exceeded token limit. Try breaking down your request into smaller parts."
                elif finish_reason == 3:
                    return "Response blocked for safety reasons. Please rephrase your request."
                elif finish_reason == 4:
                    return "Response blocked due to recitation concerns. Please try a different approach."
            
            # Try to extract content from parts
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts and len(parts) > 0:
                    text_parts = []
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    
                    if text_parts:
                        return '\n'.join(text_parts)
        
        # Fallback: try direct attribute access
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
            
            # Enhanced Gemini wrapper for smolagents compatibility
            class GeminiModel:
                def __init__(self, model_name):
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                
                def __call__(self, messages, **kwargs):
                    # Handle both string and messages format
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
                        
                        # Enhanced SafeResponse with better error handling
                        class SafeResponse:
                            def __init__(self, gemini_response):
                                self.original_response = gemini_response
                                self._content = None
                                self._text = None
                                self._extract_content()
                            
                            def _extract_content(self):
                                """Extract content safely from Gemini response"""
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
                                """Handle case where no content is returned"""
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
                        # Return error response object
                        class ErrorResponse:
                            def __init__(self, error_msg):
                                self.content = f"Gemini model error: {error_msg}"
                                self.text = self.content
                        
                        return ErrorResponse(str(e))
            
            return GeminiModel(model_name)
            
        else:  # Together AI
            os.environ["TOGETHER_API_KEY"] = api_key
            from smolagents import InferenceClientModel
            return InferenceClientModel(model_name, provider="together", max_tokens=3000)
            
    except Exception as e:
        st.error(f"Error initializing {provider} model: {str(e)}")
        return None

def generate_ai_travel_plan(model, locations_df: pd.DataFrame, trip_days: int, daily_hours: int):
    """Generate intelligent travel recommendations using AI"""
    
    if model is None:
        return "Please configure your AI model first."
    
    # Prepare location data for AI
    location_summary = locations_df.to_string(index=False)
    
    prompt = f"""
    As an expert travel planner, analyze these filming locations and create an optimal {trip_days}-day itinerary:
    
    LOCATIONS DATA:
    {location_summary}
    
    CONSTRAINTS:
    - Trip duration: {trip_days} days
    - Daily travel budget: {daily_hours} hours
    - Minimize travel time between locations
    - Group nearby locations together
    - Consider practical logistics
    
    PROVIDE:
    1. Day-by-day itinerary with specific locations
    2. Travel time estimates between locations  
    3. Best airports/transportation hubs to use
    4. Tips for efficient travel routes
    5. Must-see vs optional locations priority
    
    Format as a clear, actionable travel plan.
    """
    
    try:
        # Universal approach - try different formats
        response = None
        
        # Method 1: Try string format (Gemini, some custom models)
        try:
            response = model(prompt)
        except (AttributeError, TypeError) as e:
            if "'str' object has no attribute 'role'" in str(e):
                # Method 2: Try messages format (OpenAI, smolagents)
                messages = [{"role": "user", "content": prompt}]
                response = model(messages)
            else:
                raise e
        
        # Safe content extraction
        return safe_extract_response_text(response)
            
    except Exception as e:
        return f"Error generating AI plan: {str(e)}\n\nPlease check your AI model configuration or try a shorter prompt."

def ai_enhanced_location_search(model, query: str):
    """Use AI to find and analyze filming locations"""
    
    if model is None:
        return "Please configure your AI model first."
        
    prompt = f"""
    Find comprehensive information about filming locations worldwide. 
    Focus on: {query}
    
    For each location provide:
    1. Exact name and filming details
    2. Country and nearest major city
    3. What scenes were filmed there
    4. Best way to visit (car, train, etc.)
    
    Provide at least 10 diverse locations across different countries.
    Format as structured data that can be easily parsed.
    """
    
    try:
        # Universal approach
        response = None
        
        # Try string format first
        try:
            response = model(prompt)
        except (AttributeError, TypeError) as e:
            if "'str' object has no attribute 'role'" in str(e):
                # Try messages format
                messages = [{"role": "user", "content": prompt}]
                response = model(messages)
            else:
                raise e
        
        # Use safe extraction
        return safe_extract_response_text(response)
            
    except Exception as e:
        return f"Error in AI search: {str(e)}"

def calculate_travel_distance(coord1: Tuple[float, float], coord2: Tuple[float, float], transport_mode: str = "car") -> Dict:
    """Calculate distance and travel time between two coordinates"""
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    distance = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
    
    # Estimate travel times based on transport mode
    speeds = {"car": 70, "train": 100, "plane": 500, "mixed": 85}
    travel_time = distance / speeds.get(transport_mode, 70)
    
    return {
        "distance_km": round(distance, 2),
        "travel_time_hours": round(travel_time, 2),
        "transport_mode": transport_mode
    }

# ==================== SAMPLE DATA ====================

# Enhanced sample locations with more details
SAMPLE_LOCATIONS = {
    "Hogwarts - Alnwick Castle": (55.4180, -1.7065, "England"),
    "Platform 9¾ - King's Cross Station": (51.5308, -0.1238, "England"), 
    "Diagon Alley - Leadenhall Market": (51.5131, -0.0834, "England"),
    "Hogwarts Express - Glenfinnan Viaduct": (56.8783, -5.4318, "Scotland"),
    "Great Hall - Christ Church Oxford": (51.7501, -1.2544, "England"),
    "Hogwarts Corridors - Gloucester Cathedral": (51.8607, -2.2431, "England"),
    "Professor Snape's House - Lacock Abbey": (51.4148, -2.1187, "England"),
    "Hogwarts Courtyard - Durham Cathedral": (54.7737, -1.5755, "England"),
    "Quidditch Scenes - Cliffs of Moher": (52.9715, -9.4265, "Ireland"),
    "Forbidden Forest - Ashridge Estate": (51.7833, -0.5833, "England")
}

# ==================== MAIN APPLICATION ====================

# Main application layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🏰 Filming Locations")
    
    # AI Model initialization
    if ai_key:
        model = initialize_ai_model(ai_provider, ai_key, model_name)
        if model:
            st.success(f"✅ {ai_provider} {model_name} initialized successfully!")
    else:
        model = None
        st.warning(f"⚠️ Please enter your {ai_provider} API key to enable AI features")
    
    # Enhanced search options with custom locations
    search_option = st.radio(
        "Search Method:",
        ["🔍 Quick Search (Demo Data)", "🤖 AI-Powered Search", "🌐 Web Search + AI Analysis", "📍 Custom Locations"],
        help="Choose how to find locations"
    )

    # ENHANCED CUSTOM LOCATIONS INPUT SECTION with Nearby Places Discovery
    if search_option == "📍 Custom Locations":
        st.subheader("📍 Add Your Custom Location & Discover Nearby Places")
        st.info("💡 **New Feature**: Add a location and we'll find 10 interesting nearby places to visit!")
        
        st.write("**Add your custom destination:**")
        
        # Initialize custom locations in session state
        if 'custom_locations_input' not in st.session_state:
            st.session_state.custom_locations_input = []
        
        # Form to add custom locations WITH AUTO-GEOCODING AND NEARBY DISCOVERY
        with st.form("add_custom_location"):
            col1_form, col2_form = st.columns(2)
            with col1_form:
                custom_name = st.text_input("Location Name", placeholder="e.g., Edinburgh Castle, Central Park, Eiffel Tower")
                custom_country = st.text_input("Country (optional)", placeholder="e.g., Scotland")
            with col2_form:
                discovery_method = st.selectbox(
                    "Nearby Places Discovery",
                    ["AI + Web APIs", "AI Only", "Web APIs Only", "None"],
                    help="How to find nearby places"
                )
                radius_km = st.slider("Search Radius (km)", 5, 100, 25, help="How far to search for nearby places")
            
            custom_description = st.text_area("Description (optional)", placeholder="What makes this location special?")
            
            submitted = st.form_submit_button("➕ Add Location & Find Nearby Places")
            
            if submitted and custom_name:
                with st.spinner(f"Finding coordinates for {custom_name}..."):
                    # Auto-geocode the location
                    lat, lon, source = geocode_location_with_fallback(custom_name)
                    
                    if lat and lon:
                        # Clear previous custom locations (we want only one main location + nearby)
                        st.session_state.custom_locations_input = []
                        
                        # Add main location
                        main_location = {
                            "name": custom_name,
                            "country": custom_country or "Unknown",
                            "lat": lat,
                            "lon": lon,
                            "description": custom_description,
                            "geocoded_source": source,
                            "type": "Main Destination"
                        }
                        st.session_state.custom_locations_input.append(main_location)
                        
                        # Find nearby places based on selected method
                        nearby_places = []
                        
                        if discovery_method in ["AI + Web APIs", "AI Only"]:
                            if model:
                                with st.spinner("AI is discovering nearby attractions..."):
                                    ai_places = find_nearby_places_with_ai(model, custom_name, lat, lon, radius_km)
                                    nearby_places.extend(ai_places)
                                    st.success(f"AI found {len(ai_places)} nearby places!")
                            else:
                                st.warning("AI model not configured. Using web APIs only.")
                        
                        if discovery_method in ["AI + Web APIs", "Web APIs Only"]:
                            with st.spinner("Searching web databases for nearby places..."):
                                web_places = find_nearby_places_with_api(lat, lon, radius_km)
                                nearby_places.extend(web_places)
                                st.success(f"Web APIs found {len(web_places)} additional places!")
                        
                        # Remove duplicates and add nearby places
                        seen_names = {main_location["name"].lower()}
                        unique_nearby = []
                        
                        for place in nearby_places:
                            place_name_lower = place["name"].lower()
                            if place_name_lower not in seen_names:
                                seen_names.add(place_name_lower)
                                place["type"] = f"Nearby {place.get('type', 'Attraction')}"
                                unique_nearby.append(place)
                        
                        # Limit to 10 nearby places
                        st.session_state.custom_locations_input.extend(unique_nearby[:10])
                        
                        total_found = len(st.session_state.custom_locations_input)
                        st.success(f"✅ Added {custom_name} + {total_found-1} nearby places!")
                        st.rerun()
                    else:
                        st.error(f"❌ Could not find coordinates for '{custom_name}'. Please try a more specific location name.")
        
        # Display current custom locations
        if st.session_state.custom_locations_input:
            st.write(f"**Your Destination & Nearby Places ({len(st.session_state.custom_locations_input)} total):**")
            
            # Show main destination first
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
                    st.caption(f"Coordinates found via: {loc.get('geocoded_source', 'manual')}")
                with col_remove:
                    if st.button("🗑️", key=f"remove_main_{i}", help="Remove all locations"):
                        st.session_state.custom_locations_input = []
                        st.rerun()
            
            # Display nearby places
            if nearby_locations:
                st.markdown(f"**🌟 Nearby Places to Visit ({len(nearby_locations)}):**")
                for i, loc in enumerate(nearby_locations):
                    col_info, col_remove = st.columns([5, 1])
                    with col_info:
                        st.write(f"• **{loc['name']}** - {loc['lat']:.4f}, {loc['lon']:.4f}")
                        st.caption(f"Type: {loc.get('type', 'Attraction')} | Source: {loc.get('source', 'unknown')}")
                    with col_remove:
                        if st.button("🗑️", key=f"remove_nearby_{i}", help="Remove this place"):
                            st.session_state.custom_locations_input.pop(len(main_locations) + i)
                            st.rerun()
            
            # Button to find more nearby places
            if main_locations:
                if st.button("🔍 Find More Nearby Places"):
                    main_loc = main_locations[0]
                    with st.spinner("Finding more nearby places..."):
                        if model:
                            new_places = find_nearby_places_with_ai(model, main_loc["name"], main_loc["lat"], main_loc["lon"], 30)
                            # Add only new places
                            existing_names = {loc["name"].lower() for loc in st.session_state.custom_locations_input}
                            for place in new_places:
                                if place["name"].lower() not in existing_names:
                                    place["type"] = f"Nearby {place.get('type', 'Attraction')}"
                                    st.session_state.custom_locations_input.append(place)
                            st.rerun()

    search_query = st.text_input(
        "Search Query (optional):", 
        "Filming locations worldwide" if search_option == "📍 Custom Locations" else "Harry Potter filming locations worldwide",
        help="Customize your search query"
    )
    
    # Enhanced search button logic with custom locations support and auto-geocoding
    if st.button("🔍 Process My Destination & Nearby Places" if search_option == "📍 Custom Locations" else "🔍 Find Locations", key="search_btn"):
        with st.spinner("Processing locations..."):
            
            # Determine which locations to use
            all_locations = {}
            
            if search_option == "📍 Custom Locations":
                # Use ONLY custom locations (main + nearby places)
                if st.session_state.custom_locations_input:
                    # Convert custom locations to the expected format
                    for loc in st.session_state.custom_locations_input:
                        all_locations[loc["name"]] = (loc["lat"], loc["lon"], loc["country"])
                    
                    main_count = len([loc for loc in st.session_state.custom_locations_input if loc.get("type") == "Main Destination"])
                    nearby_count = len(st.session_state.custom_locations_input) - main_count
                    
                    st.info(f"Processing {main_count} main destination + {nearby_count} nearby places")
                else:
                    st.warning("Please add a custom location first to discover nearby places.")
                    st.stop()
                    
            else:
                # Use Harry Potter locations for other search methods
                all_locations = SAMPLE_LOCATIONS.copy()
                
                if search_option == "🔍 Quick Search (Demo Data)":
                    st.info("Using Harry Potter demo data")
                elif search_option == "🤖 AI-Powered Search":
                    if model:
                        st.info("Using AI to find and analyze Harry Potter locations...")
                        ai_results = ai_enhanced_location_search(model, search_query)
                        st.text_area("AI Search Results:", ai_results, height=200)
                    else:
                        st.error("Please configure your AI model first!")
                        
                else:  # Web Search + AI Analysis
                    if model and (serpapi_key or serper_key):
                        st.info("Combining web search with AI analysis...")
                        st.warning("Web search integration requires full smolagents setup")
                    else:
                        st.error("Please configure both AI model and search API keys!")
            
            # Process all locations (only if we have locations to process)
            if all_locations:
                location_data = []
                
                for name, (lat, lon, country) in all_locations.items():
                    travel_info = calculate_travel_distance(departure_coords, (lat, lon), transport_mode)
                    
                    # Determine location type for custom locations
                    if search_option == "📍 Custom Locations":
                        # Find the location type from session state
                        location_type = "Nearby Attraction"
                        for loc in st.session_state.custom_locations_input:
                            if loc["name"] == name:
                                if loc.get("type") == "Main Destination":
                                    location_type = "Main Destination"
                                else:
                                    location_type = loc.get("type", "Nearby Attraction")
                                break
                    else:
                        location_type = "Harry Potter"
                    
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
                # FIXED: Clean column names to prevent KeyError
                df = clean_dataframe_columns(df)
                
                st.session_state.locations_df = df
                st.session_state.departure_info = {
                    "location": departure_location,
                    "coords": departure_coords
                }
                
                if search_option == "📍 Custom Locations":
                    main_count = len([loc for loc in location_data if "Main Destination" in loc["Type"]])
                    nearby_count = len(location_data) - main_count
                    st.success(f"Processed {len(all_locations)} total locations! ({main_count} main destination, {nearby_count} nearby places)")
                else:
                    st.success(f"Found {len(all_locations)} Harry Potter filming locations!")
            else:
                st.error("No locations to process!")
    
    # Display results if available
    if 'locations_df' in st.session_state:
        st.subheader("📊 Location Data")
        
        # Show departure info
        if 'departure_info' in st.session_state:
            dep_info = st.session_state.departure_info
            st.info(f"📍 Origin: {dep_info['location']} ({dep_info['coords'][0]:.2f}, {dep_info['coords'][1]:.2f})")
        
        st.dataframe(st.session_state.locations_df, use_container_width=True)
        
        # Download button
        csv = st.session_state.locations_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"travel_locations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    st.header("📈 Trip Analytics")
    
    if 'locations_df' in st.session_state:
        df = st.session_state.locations_df
        
        # FIXED: Clean columns if not already cleaned
        if any(col != col.lower().strip() for col in df.columns):
            df = clean_dataframe_columns(df)
            st.session_state.locations_df = df
        
        # Statistics
        st.metric("Total Locations", len(df))
        st.metric("Average Distance", f"{df['distance (km)'].mean():.0f} km")
        st.metric("Closest Location", df.loc[df['distance (km)'].idxmin(), 'location'])
        
        # Show location type breakdown
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            st.write("**Location Types:**")
            for loc_type, count in type_counts.items():
                st.write(f"• {loc_type}: {count}")
        
        # Feasibility analysis
        max_daily_distance = daily_budget_hours * 70  # Assuming 70 km/h average
        feasible_locations = df[df['distance (km)'] <= max_daily_distance * trip_days]
        
        st.subheader("🎯 Trip Feasibility")
        st.write(f"**Feasible locations for {trip_days}-day trip:** {len(feasible_locations)}")
        st.write(f"**Daily travel budget:** {daily_budget_hours} hours ({max_daily_distance:.0f} km)")
        
        feasibility_score = len(feasible_locations) / len(df) if len(df) > 0 else 0
        st.progress(feasibility_score)
        st.write(f"**Feasibility Score:** {feasibility_score:.1%}")

# Enhanced map visualization with location type colors
if 'locations_df' in st.session_state:
    st.header("🗺️ Interactive Map")
    
    df = st.session_state.locations_df
    
    # FIXED: Clean columns if needed
    if any(col != col.lower().strip() for col in df.columns):
        df = clean_dataframe_columns(df)
    
    if df.empty:
        st.error("No location data available. Please search for locations first.")
    else:
        st.write(f"Showing {len(df)} locations on map")
        
        # Enhanced Plotly map with location type colors
        try:
            # Enhanced color mapping for the new location types
            color_discrete_map = {
                "harry potter": "#722F37",        # Maroon for HP locations
                "main destination": "#FF6B35",    # Orange for main destination
                "nearby attraction": "#2E8B57",   # Sea Green for nearby places
                "nearby historical": "#4169E1",   # Royal Blue for historical sites
                "nearby cultural": "#9932CC",     # Dark Violet for cultural sites
                "nearby natural": "#228B22",      # Forest Green for natural attractions
            }
            
            # Create map with different colors for location types
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                hover_name="location",
                hover_data=["country", "type", "distance (km)", "travel time (hrs)"],
                color="type" if 'type' in df.columns else "country",
                size="distance (km)",
                color_discrete_map=color_discrete_map,
                size_max=25,
                zoom=2,
                height=700,
                title="Travel Destinations Map (Auto-Geocoded Locations)",
                labels={"distance (km)": "Distance from Origin (km)"}
            )
            
            # Add origin marker
            if 'departure_info' in st.session_state:
                dep_coords = st.session_state.departure_info['coords']
                dep_location = st.session_state.departure_info['location']
                
                fig.add_trace(go.Scattermapbox(
                    lat=[dep_coords[0]],
                    lon=[dep_coords[1]],
                    mode='markers',
                    marker=dict(
                        size=20, 
                        color='red', 
                        symbol='star',
                        opacity=0.8
                    ),
                    text=[f"Origin: {dep_location}"],
                    hoverinfo='text',
                    name="🏠 Origin",
                    showlegend=True
                ))
            
            # Enhanced layout configuration
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    bearing=0,
                    pitch=0,
                    zoom=2,
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
                    x=0.01
                )
            )
            
            # Display the map
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying interactive map: {e}")
            # Fallback: Simple map
            st.subheader("📍 Simple Map View")
            map_data = df[['latitude', 'longitude', 'location']].copy()
            map_data = map_data.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            st.map(map_data)

# AI-powered itinerary planner (session state issue fixed)
if 'locations_df' in st.session_state:
    st.header("🤖 AI Travel Planner")
    
    col_ai1, col_ai2 = st.columns([2, 1])
    
    with col_ai1:
        # Initialize plan counter if not exists
        if 'plan_counter' not in st.session_state:
            st.session_state.plan_counter = 0
            
        # Generate AI Travel Plan button
        generate_plan = st.button("✨ Generate AI Travel Plan", key=f"ai_plan_{st.session_state.plan_counter}")
        
        if generate_plan:
            if model:
                with st.spinner("AI is crafting your magical itinerary..."):
                    ai_plan = generate_ai_travel_plan(
                        model, 
                        st.session_state.locations_df, 
                        trip_days, 
                        daily_budget_hours
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
    
    # Display AI-generated plan
    if 'travel_plan' in st.session_state:
        st.subheader("🗓️ Your Personalized Itinerary")
        st.markdown(st.session_state.travel_plan)
        
        # Option to regenerate with different parameters
        regenerate_plan = st.button("🔄 Regenerate Plan", key=f"regen_plan_{st.session_state.plan_counter}")
        
        if regenerate_plan:
            if model:
                with st.spinner("Creating alternative plan..."):
                    ai_plan = generate_ai_travel_plan(
                        model, 
                        st.session_state.locations_df, 
                        trip_days, 
                        daily_budget_hours
                    )
                    st.session_state.travel_plan = ai_plan
                    st.session_state.plan_counter += 1
                    st.rerun()
            else:
                st.error("Please configure your AI model first!")

# Traditional itinerary planner (fallback)
if 'locations_df' in st.session_state:
    st.header("📅 Basic Itinerary Planner")
    
    df = st.session_state.locations_df
    
    # FIXED: Clean columns if needed
    if any(col != col.lower().strip() for col in df.columns):
        df = clean_dataframe_columns(df)
    
    df = df.sort_values('distance (km)')  # Note: now lowercase
    
    # Group locations by country and proximity
    countries = df['country'].unique()  # Note: now lowercase
    
    for i, country in enumerate(countries[:trip_days]):
        with st.expander(f"Day {i+1}: {country}"):
            country_locations = df[df['country'] == country]  # Note: now lowercase
            
            st.write(f"**Suggested locations in {country}:**")
            for _, location in country_locations.head(3).iterrows():
                location_type = f" ({location['type']})" if 'type' in location else ""
                st.write(f"• **{location['location']}**{location_type} - {location['distance (km)']} km away")
            
            total_time = country_locations.head(3)['travel time (hrs)'].sum()
            st.write(f"**Estimated total travel time:** {total_time:.1f} hours")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚡ Made with magic and Streamlit ⚡</p>
    <p>🌍 Now with automatic geocoding & nearby places discovery!</p>
    <p>Remember: "It does not do to dwell on dreams and forget to live." - Dumbledore</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Deployment instructions in sidebar
with st.sidebar.expander("🚀 Deployment Guide"):
    st.markdown("""
    **To deploy this app:**
    
    **1. Streamlit Cloud:**
    - Push to GitHub
    - Connect at streamlit.io
    - Add secrets for API keys
    - Deploy in 1-click
    
    **2. Local Setup:**
    ```
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```
    
    **3. Docker:**
    ```
    docker build -t hp-planner .
    docker run -p 8501:8501 hp-planner
    ```
    
    **API Keys Needed:**
    - **OpenAI**: Get from platform.openai.com
    - **Gemini**: Get from ai.google.dev
    - **Together AI**: Get from together.ai
    - **Search**: SerpAPI or Serper
    
    **New Features:**
    - **Auto-Geocoding**: Uses free Open-Meteo & OpenStreetMap APIs
    - **Nearby Discovery**: AI + Web APIs find local attractions
    - **No Extra API Keys**: Geocoding & nearby search are free!
    
    **Choose Your Stack:**
    - **Free Tier**: Gemini Flash + Auto-Geocoding + Web APIs
    - **Performance**: OpenAI GPT-4o + Full Feature Set  
    - **Open Source**: Together AI + Free Geocoding & Discovery
    """)

if __name__ == "__main__":
    # This runs when script is executed directly
    pass
