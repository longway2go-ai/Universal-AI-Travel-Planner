# 🤖 Universal AI Travel Planner

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

> **An intelligent multi-agent system that plans your perfect trip using real-time data, AI, and smart coordination**

Transform any destination into a comprehensive travel plan with weather data, top-rated attractions, reviews, and a personalized day-by-day itinerary—all in seconds!

---

## 🌟 Features

- **🌍 Universal Location Support** - Plan trips to any city or country worldwide
- **⛅ Real-Time Weather** - Current weather conditions using Open-Meteo API
- **🏛️ Smart Attractions** - Discover top tourist spots with ratings and reviews
- **⭐ Google Ratings Integration** - Real ratings and review counts for every attraction
- **📝 Destination Reviews** - Overall ratings and recent visitor feedback
- **🤖 AI-Powered Itineraries** - Personalized day-by-day plans using Gemini or OpenAI
- **📊 Rating Analytics** - Average ratings across all suggested attractions
- **📥 Downloadable Plans** - Export your itinerary as a text file
- **🆓 Free APIs** - Core functionality works with 100% free APIs
- **👨‍💻 Beginner-Friendly** - Simple function-based code, easy to understand

---

## 🎯 What Makes This Special?

### **Multi-Agent Architecture**
This isn't just another travel app—it's an **agentic system** where specialized functions work together:

```
📍 Location Agent → ⛅ Weather Agent → 🏛️ Attractions Agent → 
⭐ Ratings Agent → ⭐ Reviews Agent → 🤖 AI Planner Agent
```

Each agent does ONE job perfectly, and the orchestrator coordinates them all!

### **Real-Time Data**
Unlike traditional systems that use static databases, this pulls **live data** from multiple sources:
- Current weather conditions
- Up-to-date attraction information
- Latest ratings and reviews
- Dynamic AI-generated itineraries

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/universal-ai-travel-planner.git
cd universal-ai-travel-planner
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run main.py
```

4. **Open your browser**
The app will automatically open at `http://localhost:8501`

---

## 🔑 API Keys Setup

### **Required: AI Provider (Free)**

Choose one:

#### **Option 1: Google Gemini (Recommended - FREE)**
1. Go to [Google AI Studio](https://ai.google.dev)
2. Click "Get API Key"
3. Create a new API key
4. Copy and paste into the app sidebar

#### **Option 2: OpenAI**
1. Go to [OpenAI Platform](https://platform.openai.com)
2. Create an account
3. Navigate to API Keys
4. Create a new key
5. Copy and paste into the app sidebar

### **Optional: Serper API (For Ratings)**

Get real Google ratings for attractions:

1. Go to [Serper.dev](https://serper.dev)
2. Sign up (free tier: 2,500 searches/month)
3. Copy your API key
4. Paste into the app sidebar

**Note:** App works without Serper, but ratings will show as "N/A"

---

## 📖 How to Use

### **Step 1: Configure**
- Select your AI provider (Gemini or OpenAI)
- Enter your API key
- Optionally add Serper key for ratings
- Choose trip duration (3-14 days)

### **Step 2: Plan**
- Enter your destination (e.g., "Paris", "Tokyo", "New York")
- Click "🚀 Create Trip Plan"
- Watch the agents work their magic!

### **Step 3: Explore**
- View location and weather information
- Browse top-rated attractions
- Read destination reviews
- Get your personalized itinerary
- Download your plan

---

## 🏗️ Architecture

### **System Design**

```
User Input
    ↓
🎯 Orchestrator (Coordinator)
    ↓
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  📍 Location Agent      ⛅ Weather Agent                     │
│       ↓                      ↓                               │
│  Get Coordinates        Get Climate Data                     │
│                                                               │
│  🏛️ Attractions Agent   ⭐ Ratings Agent                    │
│       ↓                      ↓                               │
│  Find Places           Get Reviews & Stars                   │
│                                                               │
│  ⭐ Reviews Agent       🤖 AI Planner Agent                  │
│       ↓                      ↓                               │
│  Destination Feedback   Generate Itinerary                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
    ↓
Combined Results → User Interface
```

### **Function Flow**

```python
orchestrate_trip_planning()
    ↓
find_location(destination)          # Geocoding API
    ↓
get_weather(lat, lon)               # Open-Meteo API
    ↓
find_attractions(lat, lon)          # OpenStreetMap API
    ↓
get_place_rating(place, city)       # Serper API (optional)
    ↓
get_destination_reviews(place)      # Serper API (optional)
    ↓
create_trip_plan(data, days)        # AI Model (Gemini/GPT)
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | Streamlit | Web interface |
| **Language** | Python 3.8+ | Core logic |
| **Location** | Open-Meteo Geocoding | Free coordinates lookup |
| **Weather** | Open-Meteo API | Free weather data |
| **Attractions** | OpenStreetMap Overpass | Free POI data |
| **Ratings** | Serper API | Google search results |
| **AI Models** | Gemini / OpenAI | Itinerary generation |
| **HTTP Client** | Requests | API calls |

---

## 📁 Project Structure

```
universal-ai-travel-planner/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
│
└── docs/
    ├── interview_questions.md    # 100 interview Q&A
    └── agentic_vs_rag.md        # Architecture explanation
```

---

## 🔧 Configuration

### **Supported AI Models**

**Google Gemini:**
- `gemini-2.5-flash` (Fast, efficient - Recommended)
- `gemini-2.5-pro` (More powerful)

**OpenAI:**
- `gpt-4o-mini` (Cost-effective)
- `gpt-3.5-turbo` (Fast)
- `gpt-4o` (Most capable)

### **Customization**

Edit these parameters in `main.py`:

```python
# Number of attractions to fetch
[:8]  # Line 158 - Change to get more/fewer attractions

# API timeouts
timeout=10  # Adjust for slower connections

# Temperature range for location search
({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1})  # Adjust search radius
```

---

## 📊 Example Output

**Input:** "Paris"

**Output:**
```
📍 Paris, France
   Latitude: 48.8566°
   Longitude: 2.3522°

⛅ Current Weather
   Temperature: 15°C
   Wind Speed: 12 km/h

🏛️ Top-Rated Attractions
   1. Eiffel Tower ⭐ 4.6/5 (156,789 reviews)
   2. Louvre Museum ⭐ 4.7/5 (234,567 reviews)
   3. Arc de Triomphe ⭐ 4.5/5 (98,234 reviews)
   ...

⭐ Paris Reviews
   Overall Rating: 4.8/5 ⭐
   Total Reviews: 1,234,567

📅 Your 7-Day Itinerary
   Day 1: Arrival and Eiffel Tower
   - Morning: Check into hotel, rest
   - Afternoon: Visit Eiffel Tower (4.6★)
   - Evening: Seine River cruise
   ...
```

---

## 🧪 Testing

### **Manual Testing**

Try these destinations:
- ✅ Major cities: "Paris", "Tokyo", "New York"
- ✅ Small towns: "Bruges", "Kyoto", "Santorini"
- ✅ Countries: "Iceland", "Switzerland"
- ✅ Non-English: "東京", "巴黎"

### **Edge Cases**
- ❌ Invalid locations: "Atlantis", "Xyz123"
- ⚠️ Ambiguous names: "Paris, Texas" vs "Paris, France"
- 🌐 Unicode characters: "São Paulo", "Zürich"

---

## 🐛 Troubleshooting

### **Common Issues**

**Problem:** "Location not found"
- **Solution:** Try with country name: "Paris, France"

**Problem:** "AI setup error"
- **Solution:** Verify API key is correct and has no extra spaces

**Problem:** "Timeout error"
- **Solution:** Check internet connection, try again

**Problem:** "No ratings showing (N/A)"
- **Solution:** Add Serper API key in sidebar

**Problem:** Slow performance
- **Solution:** Normal! Fetching ratings for 8 attractions takes ~15-20 seconds

---

## 📚 Learning Resources

### **Understanding the Code**

Each function is documented with:
- **Purpose**: What it does
- **Parameters**: What it needs
- **Returns**: What it gives back
- **Error handling**: How it handles failures

### **Suggested Learning Path**

1. **Start here:** Read `find_location()` - simplest function
2. **Next:** Study `get_weather()` - similar pattern
3. **Then:** Explore `find_attractions()` - more complex
4. **Advanced:** Analyze `orchestrate_trip_planning()` - coordinates everything
5. **AI:** Understand `create_trip_plan()` - prompt engineering

### **Interview Preparation**

Check `docs/interview_questions.md` for:
- 45 Conceptual questions
- 55 Technical questions
- Complete answers for each

---

## 🎓 Educational Value

### **Concepts Demonstrated**

- ✅ **Multi-agent systems** - Coordinating specialized functions
- ✅ **API integration** - Working with multiple external services
- ✅ **Error handling** - Graceful failures with try-except
- ✅ **Data flow** - Passing information between functions
- ✅ **Prompt engineering** - Crafting effective AI instructions
- ✅ **Real-time data** - Working with live information
- ✅ **Orchestration** - Managing complex workflows

### **Skills Practiced**

- Python fundamentals
- RESTful API consumption
- JSON data processing
- Asynchronous operations
- User interface design
- Error handling strategies
- System architecture

---

## 🤝 Contributing

Contributions are welcome! Here's how:

### **Adding New Features**

**Example: Hotel Recommendations**

1. Create a new function:
```python
def find_hotels(lat, lon, serper_key):
    """Find hotels near location with ratings"""
    # Your implementation
    pass
```

2. Add to orchestrator:
```python
# Step 6: Get hotels
hotels_info = find_hotels(location_info["lat"], location_info["lon"], serper_key)
results["hotels"] = hotels_info
```

3. Update UI to display results

### **Ideas for Contributions**

- 🏨 Hotel recommendations
- 🍽️ Restaurant suggestions
- 🚇 Public transport information
- 💰 Budget estimation
- 🗓️ Best time to visit
- 📸 Photo galleries
- 🎫 Ticket booking links
- 🌐 Multi-language support
- 📱 Mobile optimization
- 💾 Save/load trip plans

---

## 🔄 Roadmap

### **Version 2.0 (Planned)**
- [ ] Multi-city trip support
- [ ] User accounts and saved trips
- [ ] Collaborative trip planning
- [ ] Budget tracking
- [ ] Weather forecasts for trip dates
- [ ] Flight and hotel booking integration
- [ ] Offline mode with cached data

### **Version 2.5 (Future)**
- [ ] Mobile app (React Native)
- [ ] Social sharing features
- [ ] Travel community integration
- [ ] AI-powered photo recommendations
- [ ] Real-time price tracking
- [ ] Trip modification suggestions

---

## 📈 Performance

### **Response Times**
- Location lookup: ~1-2 seconds
- Weather fetch: ~1-2 seconds
- Attractions search: ~3-5 seconds
- Ratings (8 places): ~10-15 seconds
- AI itinerary: ~5-10 seconds
- **Total:** ~25-35 seconds

### **Optimization Opportunities**
- Parallel API calls (async/await)
- Caching frequently searched destinations
- Progressive loading (show results as available)
- Rate limiting to respect API quotas

---

## 🔒 Security & Privacy

### **Data Handling**
- ✅ No user data stored on servers
- ✅ API keys entered locally (not logged)
- ✅ All requests are HTTPS encrypted
- ✅ No tracking or analytics
- ✅ Open-source and transparent

### **Best Practices**
- Never commit API keys to version control
- Use `.env` files for local development
- Implement rate limiting for production
- Add authentication for deployed versions

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### **What This Means**
- ✅ Free to use for any purpose
- ✅ Modify and distribute
- ✅ Commercial use allowed
- ✅ Private use permitted
- ⚠️ No warranty provided

---

## 🙏 Acknowledgments

### **APIs & Services**
- [Open-Meteo](https://open-meteo.com) - Free weather and geocoding
- [OpenStreetMap](https://www.openstreetmap.org) - Open geographic data
- [Serper](https://serper.dev) - Google search API
- [Google Gemini](https://ai.google.dev) - AI model
- [OpenAI](https://openai.com) - AI model
- [Streamlit](https://streamlit.io) - Web framework

### **Inspiration**
Built as a beginner-friendly introduction to multi-agent AI systems and real-world API integration.

---

## 📞 Support & Contact

### **Get Help**
- 🐛 **Bug Reports:** Open an issue on GitHub
- 💡 **Feature Requests:** Start a discussion
- 📧 **Email:** your.email@example.com
- 💬 **Discord:** [Join our community](#)

### **FAQ**

**Q: Is this really free?**
A: Yes! All core APIs are free. Serper has a free tier too.

**Q: Can I use this commercially?**
A: Yes, MIT license allows commercial use.

**Q: How accurate is the data?**
A: Very! All data comes from reliable, real-time sources.

**Q: Can I add my own APIs?**
A: Absolutely! Just create a new agent function.

**Q: Does it work offline?**
A: No, it requires internet for API calls.

---

## ⭐ Show Your Support

If you find this project helpful:
- ⭐ Star the repository
- 🍴 Fork and contribute
- 📢 Share with friends
- 💬 Provide feedback
- 🐛 Report bugs

---

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/universal-ai-travel-planner?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/universal-ai-travel-planner?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/universal-ai-travel-planner)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/universal-ai-travel-planner)

---

## 🎯 Key Takeaway

This project demonstrates that **powerful AI applications don't need to be complex**. With:
- Simple, focused functions
- Smart coordination
- Real-time data
- AI integration

You can build production-ready applications that solve real problems!

---

<div align="center">

**Made with ❤️ for the developer community**

[⬆ Back to Top](#-universal-ai-travel-planner)

</div>
