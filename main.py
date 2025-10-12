# simple_agentic_travel.py
"""
Agentic Travel Planner - Easy to understand multi-agent system
Each agent does ONE job, orchestrator coordinates them
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Page setup
st.set_page_config(page_title="Universal AI Travel Planner", page_icon="🤖", layout="wide")

st.title("🤖 Universal AI Travel Planner")
st.write("**Easy-to-understand AI agents working together to plan your trip**")

# ==================== SIMPLE AGENT CLASSES ====================

class SearchAgent:
    """Searches the web for travel info"""
    def __init__(self, serper_key):
        self.serper_key = serper_key
        self.name = "🔍 Search Agent"
    
    def search(self, query):
        """Search web using Serper API"""
        if not self.serper_key:
            return {"error": "No API key"}
        
        try:
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": self.serper_key, "Content-Type": "application/json"}
            response = requests.post(url, json={"q": query}, headers=headers, timeout=10)
            data = response.json()
            
            results = []
            for item in data.get("organic", [])[:3]:
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet")
                })
            return {"results": results, "status": "success"}
        except Exception as e:
            return {"error": str(e)}

class LocationAgent:
    """Finds coordinates for any location"""
    def __init__(self):
        self.name = "📍 Location Agent"
    
    def find_location(self, place_name):
        """Get coordinates using free geocoding API"""
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {"name": place_name, "count": 1}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if "results" in data and data["results"]:
                result = data["results"][0]
                return {
                    "name": result["name"],
                    "country": result.get("country", ""),
                    "lat": result["latitude"],
                    "lon": result["longitude"],
                    "status": "success"
                }
            return {"error": "Location not found"}
        except Exception as e:
            return {"error": str(e)}

class WeatherAgent:
    """Gets weather for a location"""
    def __init__(self):
        self.name = "⛅ Weather Agent"
    
    def get_weather(self, lat, lon):
        """Get weather using free Open-Meteo API"""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "daily": "temperature_2m_max,temperature_2m_min"
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            current = data.get("current_weather", {})
            return {
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed"),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e)}

class AttractionsAgent:
    """Finds tourist attractions"""
    def __init__(self):
        self.name = "🏛️ Attractions Agent"
    
    def find_attractions(self, lat, lon):
        """Find nearby attractions using OpenStreetMap"""
        try:
            url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:15];
            (
              node["tourism"="attraction"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});
              node["tourism"="museum"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});
            );
            out 10;
            """
            response = requests.post(url, data=query, timeout=20)
            data = response.json()
            
            attractions = []
            for element in data.get("elements", [])[:5]:
                if "tags" in element and "name" in element["tags"]:
                    attractions.append(element["tags"]["name"])
            
            return {"attractions": attractions, "status": "success"}
        except Exception as e:
            return {"error": str(e)}

class PlannerAgent:
    """Creates the final travel plan using AI"""
    def __init__(self, ai_model):
        self.ai_model = ai_model
        self.name = "📅 Planner Agent"
    
    def create_plan(self, destination_info, days):
        """Generate itinerary using AI"""
        if not self.ai_model:
            return {"error": "No AI model configured"}
        
        prompt = f"""Create a simple {days}-day travel plan for {destination_info['name']}, {destination_info['country']}.

Weather: {destination_info.get('temperature', 'N/A')}°C
Top attractions: {', '.join(destination_info.get('attractions', [])[:3])}

Format: Day 1, Day 2, etc. with activities for each day."""
        
        try:
            response = self.ai_model(prompt)
            plan_text = response.content if hasattr(response, 'content') else str(response)
            return {"plan": plan_text, "status": "success"}
        except Exception as e:
            return {"error": str(e)}

class Orchestrator:
    """The boss agent - coordinates all other agents"""
    def __init__(self, search_agent, location_agent, weather_agent, attractions_agent, planner_agent):
        self.search = search_agent
        self.location = location_agent
        self.weather = weather_agent
        self.attractions = attractions_agent
        self.planner = planner_agent
        self.name = "🎯 Orchestrator"
    
    def plan_trip(self, destination, days):
        """Coordinate all agents to create a complete trip plan"""
        results = {"steps": []}
        
        # Step 1: Find location
        st.write(f"**Step 1:** {self.location.name} finding coordinates...")
        location_info = self.location.find_location(destination)
        results["location"] = location_info
        results["steps"].append(f"✅ Found {destination}")
        
        if "error" in location_info:
            return results
        
        # Step 2: Get weather
        st.write(f"**Step 2:** {self.weather.name} checking weather...")
        weather_info = self.weather.get_weather(location_info["lat"], location_info["lon"])
        results["weather"] = weather_info
        results["steps"].append(f"✅ Got weather data")
        
        # Step 3: Find attractions
        st.write(f"**Step 3:** {self.attractions.name} finding attractions...")
        attractions_info = self.attractions.find_attractions(location_info["lat"], location_info["lon"])
        results["attractions"] = attractions_info
        results["steps"].append(f"✅ Found {len(attractions_info.get('attractions', []))} attractions")
        
        # Step 4: Create plan with AI
        st.write(f"**Step 4:** {self.planner.name} creating itinerary...")
        combined_info = {**location_info, **weather_info, **attractions_info}
        plan_info = self.planner.create_plan(combined_info, days)
        results["plan"] = plan_info
        results["steps"].append(f"✅ Generated {days}-day plan")
        
        return results

# ==================== INITIALIZE AI MODEL ====================

def init_ai_model(provider, api_key, model_name):
    """Simple AI model initialization"""
    if not api_key:
        return None
    
    try:
        if provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            class SimpleGemini:
                def __init__(self, model):
                    self.model = genai.GenerativeModel(model)
                
                def __call__(self, prompt):
                    response = self.model.generate_content(prompt)
                    class Result:
                        def __init__(self, text):
                            self.content = text
                    return Result(response.text)
            
            return SimpleGemini(model_name)
        
        elif provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            class SimpleOpenAI:
                def __init__(self, model):
                    self.model = model
                    self.client = client
                
                def __call__(self, prompt):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    class Result:
                        def __init__(self, text):
                            self.content = text
                    return Result(response.choices[0].message.content)
            
            return SimpleOpenAI(model_name)
    except Exception as e:
        st.error(f"AI init error: {e}")
        return None

# ==================== STREAMLIT UI ====================

# Sidebar
st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("🔑 API Keys", expanded=True):
    ai_provider = st.selectbox("AI Provider", ["Google Gemini", "OpenAI"])
    
    if ai_provider == "Google Gemini":
        ai_key = st.text_input("Gemini API Key", type="password", help="Get free key at ai.google.dev")
        model = st.selectbox("Model", ["gemini-2.0-flash-exp", "gemini-1.5-flash"])
    else:
        ai_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
    
    serper_key = st.text_input("Serper API Key (optional)", type="password", help="For web search")

st.sidebar.header("🗓️ Trip Details")
trip_days = st.sidebar.slider("Days", 3, 14, 7)

# Show how the system works
st.header("🎯 How It Works")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.info("**🎯 Orchestrator**\nCoordinates all agents")
with col2:
    st.info("**📍 Location**\nFinds coordinates")
with col3:
    st.info("**⛅ Weather**\nGets climate data")
with col4:
    st.info("**🏛️ Attractions**\nFinds places to visit")
with col5:
    st.info("**📅 Planner**\nCreates itinerary")

# Main interface
st.header("✈️ Plan Your Trip")

destination = st.text_input("Where do you want to go?", placeholder="e.g., Paris, Tokyo, New York")

if st.button("🚀 Create Trip Plan", type="primary"):
    if not destination:
        st.error("Please enter a destination")
    elif not ai_key:
        st.error("Please add your AI API key in the sidebar")
    else:
        # Initialize AI
        ai_model = init_ai_model(ai_provider, ai_key, model)
        
        if ai_model:
            # Create agents
            search_agent = SearchAgent(serper_key)
            location_agent = LocationAgent()
            weather_agent = WeatherAgent()
            attractions_agent = AttractionsAgent()
            planner_agent = PlannerAgent(ai_model)
            
            # Create orchestrator
            orchestrator = Orchestrator(
                search_agent, location_agent, weather_agent, 
                attractions_agent, planner_agent
            )
            
            # Execute
            st.write("---")
            st.subheader("🤖 Agents Working...")
            
            with st.spinner("Processing..."):
                results = orchestrator.plan_trip(destination, trip_days)
            
            # Display results
            st.write("---")
            st.success("✅ Trip plan ready!")
            
            # Location info
            if "location" in results and "error" not in results["location"]:
                loc = results["location"]
                st.subheader(f"📍 {loc['name']}, {loc['country']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latitude", f"{loc['lat']:.4f}")
                with col2:
                    st.metric("Longitude", f"{loc['lon']:.4f}")
            
            # Weather info
            if "weather" in results and "error" not in results["weather"]:
                weather = results["weather"]
                st.subheader("⛅ Current Weather")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                with col2:
                    st.metric("Wind Speed", f"{weather.get('windspeed', 'N/A')} km/h")
            
            # Attractions
            if "attractions" in results and "error" not in results["attractions"]:
                attractions = results["attractions"].get("attractions", [])
                if attractions:
                    st.subheader("🏛️ Top Attractions")
                    for i, attraction in enumerate(attractions[:5], 1):
                        st.write(f"{i}. {attraction}")
            
            # Final plan
            if "plan" in results and "error" not in results["plan"]:
                st.subheader(f"📅 Your {trip_days}-Day Itinerary")
                st.write(results["plan"]["plan"])
                
                # Download
                st.download_button(
                    "📥 Download Plan",
                    results["plan"]["plan"],
                    file_name=f"trip_{destination}_{datetime.now().strftime('%Y%m%d')}.txt"
                )

# Footer
st.write("---")
st.write("**🤖 Simple Agentic System:** Each agent has ONE job. Orchestrator coordinates them all.")
st.write("**APIs Used:** Open-Meteo (FREE) • OpenStreetMap (FREE) • Serper (optional) • AI Model")

# Help section
with st.expander("❓ Need Help?"):
    st.markdown("""
    ### How to use:
    1. **Get a FREE Gemini API key** from [ai.google.dev](https://ai.google.dev)
    2. **Paste it in the sidebar**
    3. **Enter a destination** (any city or country)
    4. **Click "Create Trip Plan"**
    
    ### What happens:
    - 📍 Location Agent finds the coordinates
    - ⛅ Weather Agent gets current weather
    - 🏛️ Attractions Agent finds places to visit
    - 📅 Planner Agent creates your itinerary using AI
    - 🎯 Orchestrator coordinates everything
    
    ### APIs:
    - **Free:** Open-Meteo, OpenStreetMap
    - **Optional:** Serper for web search ($5/month)
    - **AI:** Gemini (FREE tier) or OpenAI
    """)

# Requirements
with st.sidebar.expander("📦 Install"):
    st.code("""pip install streamlit
pip install requests
pip install google-generativeai
pip install openai""")
