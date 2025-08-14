# ⚡ Agentic AI Travel Planner
An intelligent travel planning application that helps you discover Harry Potter filming locations worldwide and find interesting nearby places to visit. Features automatic geocoding, AI-powered trip planning, and interactive maps.


Quick Search: Pre-loaded Harry Potter filming locations

AI-Powered Search: Use AI models to find and analyze locations

Custom Locations: Add your own destinations with automatic coordinate detection

Nearby Places Discovery: Automatically find 10+ interesting places near your custom locations

## 🌍 Automatic Geocoding
No API Keys Required: Uses free Open-Meteo and OpenStreetMap APIs

Smart Fallbacks: Multiple geocoding services for reliability

Global Coverage: Works worldwide for any location name

## 🤖 AI Integration
Multiple AI Providers: OpenAI GPT, Google Gemini, Together AI

Intelligent Trip Planning: Generate optimized itineraries

Smart Location Analysis: AI-curated nearby attractions

Robust Error Handling: Safe response extraction and fallbacks

## 🗺️ Interactive Visualizations
Enhanced Maps: Plotly-powered interactive maps with custom markers

Location Types: Visual distinction between main destinations and nearby places

Origin Tracking: Shows your departure point and travel distances

Fallback Options: Simple maps if interactive features fail

## 📊 Trip Analytics
Distance Calculations: Haversine formula for accurate distances

Travel Time Estimates: Based on transport mode (car, train, mixed)

Feasibility Analysis: Smart recommendations based on trip duration

Budget Planning: Daily travel hour budgeting

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
Code Structure
Modular Design: Separate functions for geocoding, AI, visualization
Error Handling: Comprehensive try-catch blocks and fallbacks
State Management: Proper Streamlit session state usage
Column Cleaning: Consistent DataFrame column naming

🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📝 License
```This project is licensed under the MIT License - see the LICENSE file for details.
```
## 🙏 Acknowledgments
Harry Potter: Warner Bros. and J.K. Rowling for the magical universe
Streamlit: For the amazing web app framework
OpenStreetMap: For free geocoding and mapping data
Open-Meteo: For reliable geocoding services
AI Providers: OpenAI, Google, and Together AI for language models

🐛 Known Issues & Solutions
Common Errors
KeyError: 'country': Fixed with automatic column cleaning
Gemini API errors: Handled with safe response extraction
Geocoding failures: Multiple fallback APIs implemented

Performance Tips
Use custom locations mode for faster processing
Enable AI features only when needed
Download results as CSV for offline analysis

📞 Support
Issues: GitHub Issues
Discussions: GitHub Discussions
Documentation: See inline code comments and docstrings

"It does not do to dwell on dreams and forget to live." - Albus Dumbledore

⚡ Made with magic and Streamlit ⚡
