# Universal AI Travel Planner

🌍 An advanced AI-powered travel planner web application built with Streamlit. Discover amazing destinations worldwide, auto-geocode any location, find nearby attractions, get user reviews via Serper API, and generate detailed AI-crafted travel itineraries with smart analytics and interactive maps.

---

## Features

- **Auto Geocoding:** Converts location names to geographic coordinates automatically using free services.
- **Nearby Places Discovery:** Finds diverse attractions near your destinations with AI and OpenStreetMap APIs.
- **User Reviews Integration:** Fetches real user reviews for places from the Serper API to enhance trip insights.
- **AI-Powered Itineraries:** Generates detailed day-by-day travel plans including highlights and user reviews.
- **Interactive Maps:** Visualize your travel destinations on an interactive map with custom styling.
- **Budget & Feasibility Analysis:** Includes trip feasibility and budget estimations based on your parameters.
- **Multiple Travel Themes:** Explore preset travel themes like Historical Europe, Tropical Paradise, Adventure Mountains, etc.
- **Multi-Provider AI Support:** Use OpenAI, Google Gemini, or Together AI models to power AI features.
- **Download Options:** Export your destinations and itineraries as CSV or text files.

---


Open the web browser at `http://localhost:8501`.

---

## Usage

- Select your departure city or enter a custom location.
- Choose your travel theme or create a custom destination list.
- Discover nearby attractions using AI or web APIs.
- Generate a detailed, AI-powered travel itinerary with highlights and real user reviews.
- Visualize your trip on the interactive map.
- Download your itinerary and destination data for offline use.

---

## Configuration

- **Trip Duration:** Set how many days your trip will last.
- **Daily Travel Budget:** Control how many hours per day you want to spend traveling.
- **Budget Range:** Choose between Budget, Mid-range, Luxury, or Ultra-luxury tiers.
- **Transport Mode:** Specify your preferred transport (car, train, plane, or mixed).
- **AI Provider:** Select between OpenAI, Google Gemini, or Together AI for AI-powered recommendations.
- **Serper API Key:** Required to enable fetching user reviews for destinations.

---

## Project Structure

- `streamlit_app.py` — Main Streamlit application and UI.
- Helper functions for geocoding, nearby places discovery, AI integration, and Serper API reviews.
- Requirements file with all dependencies listed.

---

## Notes & Tips

- Serper API key is optional but highly recommended to get authentic user reviews.
- Geocoding uses free APIs with rate limits, so heavy usage might encounter delays.
- AI-generated itineraries depend on the selected AI model and prompt design.
- Always validate travel plans and bookings independently.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Required Python packages (see `requirements.txt`)


## 🚀 Quick Start
```bash
git clone https://github.com/yourusername/harry-potter-travel-planner.git
cd harry-potter-travel-planner
Install dependencies
```
```
bash
pip install -r requirements.txt
Run the application
```
`

📦 Dependencies
```text
# Core web framework
streamlit>=1.48.0

# Data manipulation and analysis
pandas>=2.0.0
numpy>=1.24.0

# Visualization
plotly>=5.15.0

# AI/ML libraries
openai>=1.0.0
google-generativeai>=0.8.0
smolagents>=1.21.1

# HTTP requests and API calls
requests>=2.31.0
httpx>=0.28.0

# Additional utilities
python-dateutil>=2.8.2
google-search-results>=2.4.2
```
## 🔧 Configuration

AI Models (Optional)
To enable AI-powered features, add your API keys in the sidebar:

OpenAI: Get from platform.openai.com
Google Gemini: Get from ai.google.dev
Together AI: Get from together.ai

Search APIs (Optional)
For enhanced web search capabilities:
SerpAPI: Get from serpapi.com
Serper: Get from serper.dev

## 💡 Usage
1. Set Your Origin
Choose your departure location using one of three methods:
Select from 15+ popular cities
Enter any location name (auto-geocoded)
Input exact coordinates

2. Choose Discovery Method
🔍 Quick Search: Use pre-loaded Harry Potter locations
🤖 AI-Powered: Let AI find and analyze locations
📍 Custom Locations: Add your own destinations + discover nearby places

3. Plan Your Trip
Configure trip duration and daily travel budget
Generate AI-powered itineraries
View interactive maps and analytics
Download location data as CSV

4. Explore Nearby Places
When using custom locations:
Automatically discovers 10+ nearby attractions
Uses both AI and web APIs for comprehensive results
Categorizes places by type (historical, cultural, natural, etc.)

## 🏗️ Architecture
```text
harry-potter-travel-planner/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── README.md                # This file
└── .gitignore              # Git ignore rules
```
Key Components
Geocoding Engine: Free APIs (Open-Meteo + OpenStreetMap)
AI Integration: Universal compatibility layer for multiple providers
Nearby Discovery: Combines AI curation with OpenStreetMap data
Trip Calculator: Haversine distance + travel time estimation
Visualization Engine: Plotly maps with fallback options

## 🌟 Advanced Features
Custom Location Discovery
python
### Add a location and automatically find nearby places
location = "Edinburgh Castle"
nearby_places = find_nearby_places_with_ai(model, location, lat, lon, radius_km=25)
AI-Powered Itinerary Generation
python
### Generate optimized travel plans
itinerary = generate_ai_travel_plan(model, locations_df, trip_days=5, daily_hours=6)
Smart Error Handling
Graceful API fallbacks

Safe response extraction for all AI models

Consistent column naming to prevent KeyErrors

## 🚀 Deployment
Streamlit Cloud (Recommended)
Push code to GitHub
Connect repository at streamlit.io
Add API keys in Streamlit secrets
Deploy with one click
Local Docker
bash
# Build image
docker build -t hp-planner .

# Run container
docker run -p 8501:8501 hp-planner
Environment Variables
bash
# Optional AI API Keys
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
TOGETHER_API_KEY=your_together_key

# Optional Search API Keys
SERPAPI_KEY=your_serpapi_key
SERPER_KEY=your_serper_key
🎯 Use Cases
🎬 Harry Potter Fans
Discover all filming locations worldwide
Plan themed travel itineraries
Find nearby magical experiences

🌍 General Travel Planning
🎯 Use Cases
🎬 Harry Potter Fans
Discover all filming locations worldwide
Plan themed travel itineraries
Find nearby magical experiences

🌍 General Travel Planning
Add custom destinations anywhere in the world
Discover hidden gems near your locations
Generate optimized travel routes

🏢 Travel Agencies
Create themed travel packages
Automatic itinerary generation
Interactive client presentations

📚 Educational Projects
Geography and mapping exercises
AI integration demonstrations
Data visualization examples

🔄 API Integrations
Free Services (No Keys Required)
Open-Meteo Geocoding: Location coordinate lookup
OpenStreetMap Nominatim: Alternative geocoding
Overpass API: Nearby attractions discovery

AI Services (Optional)
OpenAI GPT: Advanced language understanding
Google Gemini: Multimodal AI capabilities
Together AI: Open-source model hosting
Search Services (Optional)
SerpAPI: Web search results

Serper: Alternative search API

🛠️ Development
Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
# Install dependencies
```
pip install -r requirements.txt
```
# Run in development mode
```
streamlit run streamlit_app.py --server.runOnSave true
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- Open-Meteo and Nominatim OpenStreetMap for free geocoding APIs.
- Serper API for real-time user reviews integration.
- Plotly for creating beautiful interactive maps.
- Streamlit for an easy-to-use web framework.

---

## Contact

For questions, issues, or contributions, please open an issue or submit a pull request.

Happy travels! ✈️🗺️🌟


⚡ Made with magic and Streamlit ⚡

