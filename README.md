# ✈️ Universal AI Travel Planner

A modern, intelligent travel planning application powered by AI, featuring smart ticket parsing, budget planning, hotel & restaurant recommendations, and FAISS vector database for enhanced travel insights.

![Travel Planner](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0081CB?style=for-the-badge&logo=meta&logoColor=white)

## 🌟 Features

### 🎫 Smart Ticket Parser
- **PDF & Image Support**: Upload flight tickets in PDF or image format
- **Auto-Extraction**: Automatically extracts destination, dates, and flight codes
- **OCR Technology**: Uses Tesseract OCR for image-based tickets
- **Trip Duration Calculator**: Automatically calculates trip days from ticket dates

### 🤖 AI-Powered Itinerary Generation
- **Multiple AI Providers**: 
  - Google Gemini (gemini-2.5-flash, gemini-2.5-pro)
  - OpenAI (gpt-4o-mini, gpt-3.5-turbo)
- **Personalized Plans**: Day-by-day itineraries tailored to your preferences
- **Budget-Aware**: Considers your budget constraints
- **RAG-Enhanced**: Uses vector database for context-aware recommendations

### 🏛️ Attraction Discovery
- **Interactive Carousel**: Browse attractions one by one with beautiful cards
- **Real-Time Data**: Fetches attractions from OpenStreetMap
- **Ratings & Rankings**: Each attraction shows rating (3.7-4.5) and rank
- **Image Gallery**: Visual preview of each attraction

### 🏨 Hotel Recommendations
- **Side-by-Side Layout**: Hotels displayed in organized cards
- **Star Ratings**: Shows hotel class (3-5 stars) and user ratings
- **Contact Information**: Addresses, phone numbers, and websites
- **Visual Cards**: Gradient-styled cards with images

### 🍽️ Restaurant Finder
- **Cuisine Types**: Displays restaurant categories
- **Location-Based**: Finds nearby restaurants using geolocation
- **Complete Details**: Ratings, addresses, and contact info
- **Side-by-Side View**: Restaurants displayed alongside hotels

### 💰 Budget Planning
- **Flexible Budgeting**: Set total or daily budget
- **Smart Breakdown**: Automatic calculation of:
  - Daily budget
  - Meal costs (estimated)
  - Accommodation costs (estimated)
- **Visual Metrics**: Color-coded budget cards

### 🗄️ FAISS Vector Database
- **Semantic Search**: Find attractions, hotels, and restaurants by meaning
- **Persistent Storage**: All data saved locally in `./faiss_db/`
- **RAG Integration**: Enhances AI responses with stored knowledge
- **Fast Retrieval**: Lightweight and efficient vector search

### 🌤️ Weather Integration
- **Real-Time Weather**: Current temperature and wind speed
- **Location-Aware**: Automatic weather fetching for destination

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Tesseract OCR (optional, for image ticket parsing)

### Step 1: Clone the Repository
```

## 🎯 Usage Guide

### 1️⃣ Configure API Keys (Sidebar)
1. Select your AI provider (Google Gemini or OpenAI)
2. Enter your API key
3. Choose your preferred model

**Get Free API Keys:**
- **Gemini**: [ai.google.dev](https://ai.google.dev) (Free tier available)
- **OpenAI**: [platform.openai.com](https://platform.openai.com)

### 2️⃣ Upload Tickets (Optional)
- Upload outbound and return tickets (PDF or images)
- System automatically extracts destination and dates
- Trip duration calculated automatically

### 3️⃣ Enter Trip Details
- **Destination**: City or location name
- **Trip Days**: Number of days (auto-filled if tickets uploaded)
- **Budget**: Total or daily budget in USD
- **Budget Type**: Choose "Total Budget" or "Daily Budget"

### 4️⃣ Generate Itinerary
Click **"Generate Travel Plan"** and watch the magic happen:
- 📍 Location detection
- ⛅ Weather fetching
- 🏛️ Attraction discovery
- 🏨 Hotel recommendations
- 🍽️ Restaurant finder
- 🤖 AI itinerary generation

### 5️⃣ Explore Results
- Browse attractions in carousel tabs
- Check hotels and restaurants side-by-side
- Review day-by-day itinerary
- Download complete plan as text file

### 6️⃣ Search Vector Database
Use the search tool at the bottom to query stored information:
- "best hotels in Paris"
- "Italian restaurants in Rome"
- "top attractions in Tokyo"

## 🏗️ Project Structure

```
travel-planner/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── packages.txt           # System packages (for Streamlit Cloud)
├── faiss_db/              # FAISS vector database storage
│   ├── index.faiss        # Vector index
│   └── documents.pkl      # Document metadata
└── chroma_db/             # Legacy (not used)
```

## 🎨 UI Features

### Modern Design Elements
- **Gradient Cards**: Beautiful gradient backgrounds for all sections
- **Color-Coded Metrics**: Different colors for location, weather, etc.
- **Responsive Layout**: Adapts to different screen sizes
- **Hover Effects**: Interactive image zoom and button animations
- **Tab Navigation**: Easy browsing through attractions
- **Collapsible Days**: Expandable day-by-day itinerary sections

### Accessibility
- High contrast text (white on dark backgrounds)
- Clear visual hierarchy
- Readable font sizes
- Emoji icons for quick recognition

## 🔧 Configuration

### Custom Styling
The app includes custom CSS for enhanced readability:
- Rounded corners on all elements
- Smooth transitions and hover effects
- Gradient backgrounds
- Enhanced shadows
- Better spacing and typography

### Database Management
- **View Stats**: Check stored documents in sidebar
- **Clear Database**: Reset all stored data with one click
- **Automatic Persistence**: Data saved automatically to disk

## 📊 Data Sources

- **Geocoding**: Open-Meteo Geocoding API
- **Weather**: Open-Meteo Weather API
- **Attractions/Hotels/Restaurants**: OpenStreetMap Overpass API
- **Images**: Lorem Picsum (placeholder images)
- **AI**: Google Gemini / OpenAI GPT

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `packages.txt` with:
   ```
   tesseract-ocr
   ```
5. Deploy!

### Local Development
```bash
streamlit run app.py
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Known Issues & Limitations

- OCR accuracy depends on ticket image quality
- Free API rate limits apply for Gemini/OpenAI
- OpenStreetMap data availability varies by location
- Images are placeholders (Lorem Picsum)
- Some locations may have limited hotel/restaurant data

## 🔮 Future Enhancements

- [ ] Real attraction images via Unsplash/Pexels API
- [ ] Multi-language support
- [ ] Currency conversion
- [ ] Flight price comparison
- [ ] Booking integration
- [ ] Social sharing features
- [ ] Mobile app version
- [ ] Offline mode
- [ ] User accounts and saved trips
- [ ] Collaborative trip planning

## 💡 Tips & Tricks

1. **Best Results**: Use descriptive destination names (e.g., "Paris, France" vs "Paris")
2. **Budget Planning**: Set realistic budgets for better recommendations
3. **Ticket Upload**: Ensure tickets have clear, readable text
4. **Database Search**: Use natural language queries for better results
5. **API Keys**: Use Gemini for free tier, OpenAI for advanced features

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: @yourusername

## 🙏 Acknowledgments

- **Streamlit** - Amazing web framework
- **OpenStreetMap** - Free geographic data
- **Open-Meteo** - Free weather API
- **FAISS** - Efficient vector search
- **Google & OpenAI** - Powerful AI models

## ⭐ Show Your Support

If you like this project, please give it a ⭐ on GitHub!

---

**Made with ❤️ for travelers worldwide**

🌍 Happy Travels! ✈️bash
git clone https://github.com/yourusername/travel-planner.git
cd travel-planner
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Tesseract OCR (Optional)

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Step 4: Run the Application
```bash
streamlit run app.py
```

## 📋 Requirements

```
streamlit>=1.28.0
requests>=2.31.0
PyPDF2>=3.0.0
Pillow>=10.0.0
pytesseract>=0.3.10
faiss-cpu>=1.7.4
numpy>=1.24.0
google-generativeai>=0.3.0
openai>=1.3.0
