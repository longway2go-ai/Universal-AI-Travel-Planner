"""
Smart Travel Planner - Beginner-friendly with simple functions
Each function does ONE job, main orchestrator function coordinates them
"""
import streamlit as st
import requests
from datetime import datetime
import re

# Page setup
st.set_page_config(page_title="Universal AI Travel Planner", page_icon="🤖", layout="wide")

st.title("🤖 Universal AI Travel Planner")
st.write("**Simple functions working together to plan your trip**")

# ==================== SIMPLE FUNCTIONS ====================

def search_web(query, serper_key):
    """Search the web for travel info"""
    if not serper_key:
        return {"error": "No API key"}
    
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
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


def find_location(place_name):
    """Get coordinates for any location using free geocoding"""
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


def get_weather(lat, lon):
    """Get current weather for coordinates"""
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


def get_place_rating(place_name, city_name, serper_key):
    """Get rating for a specific place"""
    if not serper_key:
        return {"rating": "N/A", "reviews": 0}
    
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
        search_query = f"{place_name} {city_name} rating"
        payload = {"q": search_query}
        
        response = requests.post(url, json=payload, headers=headers, timeout=8)
        data = response.json()
        
        # Try knowledge graph first
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            rating = kg.get("rating")
            reviews = kg.get("ratingCount", 0)
            if rating:
                return {"rating": float(rating), "reviews": reviews}
        
        # Fallback: search snippets for rating
        for result in data.get("organic", [])[:3]:
            snippet = result.get("snippet", "").lower()
            rating_match = re.search(r'(\d+\.?\d*)\s*(?:stars?|/5|★)', snippet)
            if rating_match:
                return {"rating": float(rating_match.group(1)), "reviews": "N/A"}
        
        return {"rating": "N/A", "reviews": 0}
    except:
        return {"rating": "N/A", "reviews": 0}


def find_attractions(lat, lon, city_name, serper_key):
    """Find tourist attractions near coordinates with ratings"""
    try:
        # Get attractions from OpenStreetMap
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
        for element in data.get("elements", [])[:8]:
            if "tags" in element and "name" in element["tags"]:
                place_name = element["tags"]["name"]
                
                # Get rating for this place
                rating_info = get_place_rating(place_name, city_name, serper_key)
                
                attractions.append({
                    "name": place_name,
                    "rating": rating_info.get("rating", "N/A"),
                    "reviews_count": rating_info.get("reviews", 0),
                    "type": element["tags"].get("tourism", "attraction")
                })
        
        return {"attractions": attractions, "status": "success"}
    except Exception as e:
        return {"error": str(e)}


def get_destination_reviews(place_name, serper_key):
    """Get comprehensive reviews for destination"""
    if not serper_key:
        return {"error": "No API key"}
    
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
        payload = {"q": f"{place_name} reviews ratings"}
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        
        result = {
            "overall_rating": "N/A",
            "reviews_count": 0,
            "reviews": [],
            "status": "success"
        }
        
        # Get data from knowledge graph
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            result["overall_rating"] = kg.get("rating", "N/A")
            result["reviews_count"] = kg.get("ratingCount", 0)
            
            # Get individual reviews
            if "reviews" in kg:
                for review in kg["reviews"][:5]:
                    result["reviews"].append({
                        "author": review.get("author", "Anonymous"),
                        "rating": review.get("rating", "N/A"),
                        "text": review.get("text", "")[:200],
                        "date": review.get("date", "")
                    })
        
        return result
    except Exception as e:
        return {"error": str(e)}


def calculate_average_rating(attractions):
    """Calculate average rating from attractions list"""
    ratings = [a.get('rating') for a in attractions if a.get('rating') != 'N/A']
    if not ratings:
        return 0
    return sum(ratings) / len(ratings)


def create_trip_plan(destination_info, days, ai_model):
    """Generate travel itinerary using AI"""
    if not ai_model:
        return {"error": "No AI model configured"}
    
    # Extract attraction names from the attractions list
    attractions_list = destination_info.get('attractions', [])
    if attractions_list and isinstance(attractions_list[0], dict):
        # If attractions is a list of dicts, extract names
        attraction_names = [a.get('name', '') for a in attractions_list[:3]]
    else:
        # If it's already a list of strings
        attraction_names = attractions_list[:3]
    
    prompt = f"""Create a simple {days}-day travel plan for {destination_info['name']}, {destination_info['country']}.

Weather: {destination_info.get('temperature', 'N/A')}°C
Top attractions: {', '.join(attraction_names) if attraction_names else 'To be discovered'}

Format: Day 1, Day 2, etc. with activities for each day."""
    
    try:
        response = ai_model(prompt)
        plan_text = response.content if hasattr(response, 'content') else str(response)
        return {"plan": plan_text, "status": "success"}
    except Exception as e:
        return {"error": str(e)}


def orchestrate_trip_planning(destination, days, serper_key, ai_model):
    """Main function that coordinates all steps"""
    results = {"steps": []}
    
    # Step 1: Find location
    st.write("**Step 1:** 📍 Finding coordinates...")
    location_info = find_location(destination)
    results["location"] = location_info
    results["steps"].append(f"✅ Found {destination}")
    
    if "error" in location_info:
        return results
    
    # Step 2: Get weather
    st.write("**Step 2:** ⛅ Checking weather...")
    weather_info = get_weather(location_info["lat"], location_info["lon"])
    results["weather"] = weather_info
    results["steps"].append(f"✅ Got weather data")
    
    # Step 3: Find attractions with ratings
    st.write("**Step 3:** 🏛️ Finding top-rated attractions...")
    attractions_info = find_attractions(
        location_info["lat"], 
        location_info["lon"],
        location_info["name"],
        serper_key
    )
    results["attractions"] = attractions_info
    
    if "error" not in attractions_info:
        num_attractions = len(attractions_info.get('attractions', []))
        avg_rating = calculate_average_rating(attractions_info.get('attractions', []))
        results["steps"].append(f"✅ Found {num_attractions} attractions (avg ⭐{avg_rating:.1f})")
    
    # Step 4: Get destination reviews
    st.write("**Step 4:** ⭐ Fetching destination reviews...")
    reviews_info = get_destination_reviews(destination, serper_key)
    results["reviews"] = reviews_info
    results["steps"].append(f"✅ Got destination reviews")
    
    # Step 5: Create AI-powered plan
    st.write("**Step 5:** 📅 Creating personalized itinerary...")
    combined_info = {**location_info, **weather_info, **attractions_info}
    plan_info = create_trip_plan(combined_info, days, ai_model)
    results["plan"] = plan_info
    results["steps"].append(f"✅ Generated {days}-day plan")
    
    return results


# ==================== AI MODEL SETUP ====================

def setup_ai_model(provider, api_key, model_name):
    """Initialize AI model based on provider"""
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
        st.error(f"AI setup error: {e}")
        return None


# ==================== STREAMLIT UI ====================

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("🔑 API Keys", expanded=True):
    ai_provider = st.selectbox("AI Provider", ["Google Gemini", "OpenAI"])
    
    if ai_provider == "Google Gemini":
        ai_key = st.text_input("Gemini API Key", type="password", help="Get free key at ai.google.dev")
        model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    else:
        ai_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"])
    
    serper_key = st.text_input("Serper API Key (optional)", type="password", help="For ratings")

st.sidebar.header("🗓️ Trip Details")
trip_days = st.sidebar.slider("Days", 3, 14, 7)

# Show workflow
st.header("🎯 How It Works")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.info("**📍 Location**\nFind coordinates")
with col2:
    st.info("**⛅ Weather**\nGet climate")
with col3:
    st.info("**🏛️ Attractions**\nFind places")
with col4:
    st.info("**⭐ Ratings**\nGet reviews")
with col5:
    st.info("**📅 Plan**\nCreate itinerary")

# Main interface
st.header("✈️ Plan Your Trip")

destination = st.text_input("Where do you want to go?", placeholder="e.g., Paris, Tokyo, New York")

if st.button("🚀 Create Trip Plan", type="primary"):
    if not destination:
        st.error("Please enter a destination")
    elif not ai_key:
        st.error("Please add your AI API key in the sidebar")
    else:
        # Setup AI model
        ai_model = setup_ai_model(ai_provider, ai_key, model)
        
        if ai_model:
            st.write("---")
            st.subheader("🤖 Processing Your Trip...")
            
            with st.spinner("Working on it..."):
                results = orchestrate_trip_planning(destination, trip_days, serper_key, ai_model)
            
            # Display results
            st.write("---")
            st.success("✅ Trip plan ready!")
            
            # Location
            if "location" in results and "error" not in results["location"]:
                loc = results["location"]
                st.subheader(f"📍 {loc['name']}, {loc['country']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latitude", f"{loc['lat']:.4f}")
                with col2:
                    st.metric("Longitude", f"{loc['lon']:.4f}")
            
            # Weather
            if "weather" in results and "error" not in results["weather"]:
                weather = results["weather"]
                st.subheader("⛅ Current Weather")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                with col2:
                    st.metric("Wind Speed", f"{weather.get('windspeed', 'N/A')} km/h")
            
            # Attractions with ratings
            if "attractions" in results and "error" not in results["attractions"]:
                attractions = results["attractions"].get("attractions", [])
                if attractions:
                    st.subheader("🏛️ Top-Rated Attractions")
                    
                    # Sort by rating
                    sorted_attractions = sorted(
                        attractions, 
                        key=lambda x: x.get('rating') if x.get('rating') != 'N/A' else 0, 
                        reverse=True
                    )
                    
                    for i, attraction in enumerate(sorted_attractions[:8], 1):
                        col_name, col_rating = st.columns([3, 1])
                        
                        with col_name:
                            st.write(f"**{i}. {attraction['name']}**")
                            st.caption(f"Type: {attraction.get('type', 'attraction')}")
                        
                        with col_rating:
                            rating = attraction.get('rating')
                            reviews = attraction.get('reviews_count', 0)
                            
                            if rating != 'N/A':
                                stars = "⭐" * int(rating)
                                st.metric("Rating", f"{rating}/5")
                                st.caption(f"{stars}")
                                if reviews > 0:
                                    st.caption(f"{reviews:,} reviews")
                            else:
                                st.metric("Rating", "N/A")
                    
                    # Average rating
                    avg_rating = calculate_average_rating(attractions)
                    if avg_rating > 0:
                        st.info(f"📊 Average rating: ⭐ {avg_rating:.1f}/5")
            
            # Destination reviews
            if "reviews" in results and "error" not in results["reviews"]:
                reviews_data = results["reviews"]
                st.subheader(f"⭐ {destination} Reviews")
                
                if reviews_data.get("overall_rating") != "N/A":
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.metric("Overall Rating", f"{reviews_data['overall_rating']}/5 ⭐")
                    with col_r2:
                        st.metric("Total Reviews", f"{reviews_data.get('reviews_count', 0):,}")
                
                # Individual reviews
                reviews_list = reviews_data.get("reviews", [])
                if reviews_list:
                    st.write("**Recent Reviews:**")
                    for review in reviews_list[:3]:
                        with st.expander(f"⭐ {review.get('rating', 'N/A')}/5 - {review.get('author', 'Anonymous')}"):
                            st.write(review.get('text', 'No text'))
                            if review.get('date'):
                                st.caption(f"Date: {review['date']}")
            
            # Final itinerary
            if "plan" in results and "error" not in results["plan"]:
                st.subheader(f"📅 Your {trip_days}-Day Itinerary")
                st.write(results["plan"]["plan"])
                
                # Download button
                st.download_button(
                    "📥 Download Plan",
                    results["plan"]["plan"],
                    file_name=f"trip_{destination}_{datetime.now().strftime('%Y%m%d')}.txt"
                )

# Footer
st.write("---")
st.write("**🎯 Function-Based System:** Simple functions, easy to understand!")
st.write("**APIs:** Open-Meteo (FREE) • OpenStreetMap (FREE) • Serper (ratings)")

# Help
with st.expander("❓ Help & Setup"):
    st.markdown("""
    ### Quick Start:
    1. Get **FREE Gemini API** from [ai.google.dev](https://ai.google.dev)
    2. Get **Serper API** from [serper.dev](https://serper.dev) (optional, for ratings)
    3. Paste keys in sidebar
    4. Enter destination and click "Create Trip Plan"
    
    ### Functions Explained:
    - `find_location()` - Gets coordinates
    - `get_weather()` - Fetches weather
    - `find_attractions()` - Finds places
    - `get_place_rating()` - Gets ratings
    - `get_destination_reviews()` - Gets reviews
    - `create_trip_plan()` - Generates itinerary
    - `orchestrate_trip_planning()` - Coordinates everything
    
    ### Install:
    ```bash
    pip install streamlit requests google-generativeai openai
    ```
    """)
