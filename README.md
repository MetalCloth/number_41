# Chinook RAG Q&A System 🎵

A real-world RAG (Retrieval-Augmented Generation) system that combines structured database queries with semantic search over unstructured data. Built with LangChain, LangGraph, and Streamlit.

## 🎯 What This Does

This system lets you ask natural language questions about the Chinook music store database AND customer reviews. The AI agent intelligently decides whether to:
- Query the PostgreSQL database for factual data (sales, customers, invoices)
- Search through customer reviews using semantic similarity
- Use both sources to give you comprehensive answers

**Example questions:**
- "What are the top 5 selling albums?"
- "Show me complaints about shipping"
- "Which customers spent the most money and what do they think about our service?"

## 🏗️ Architecture

```
User Question
    ↓
ReAct Agent (Kimi-K2)
    ↓
[Decision Making]
    ↓
├─→ SQL Query Tool (Structured Data)
├─→ Schema Tool (DB Understanding)  
└─→ Review Search Tool (Unstructured Data)
    ↓
FAISS Vector Store ← Customer Reviews
PostgreSQL ← Chinook Database
    ↓
Final Answer
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- PostgreSQL (running locally or Docker)
- Groq API Key

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd chinook-rag-system
```

### 2. Install dependencies
```bash
pip install streamlit langchain langchain-community langchain-groq langchain-huggingface
pip install faiss-cpu sentence-transformers psycopg2-binary python-dotenv
```

### 3. Load Chinook Database
```bash
# Download from: https://github.com/lerocha/chinook-database
# Then load into PostgreSQL
psql -U postgres -d chinook -f Chinook_PostgreSQL.sql
```

**Note:** If you hit encoding issues like I did, force UTF-8:
```bash
psql -U postgres --set client_encoding=UTF8 -d chinook -f Chinook_PostgreSQL.sql
```

**If you get UTF-16LE encoding warnings:**
Git might detect some files as UTF-16LE and offer to transcode them to UTF-8 on commit. This is fine! UTF-8 is the standard and will work better across different systems. Just let Git do the conversion.

### 4. Prepare Your Review Data
Create a JSON file with customer reviews (example structure):
```json
[
  {"review": "Great album, fast shipping!", "rating": 5},
  {"review": "Sound quality could be better", "rating": 3}
]
```

### 5. Build FAISS Index
Run the embedding script to create your vector store:
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load your review data and create embeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=FAISS.from_documents(documents, embeddings)
vector_store.save_local("json_faiss_index")
```

### 6. Configure Environment
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 7. Run the App
```bash
streamlit run app.py
```

## 🎬 Demo Video


https://github.com/user-attachments/assets/9deed786-ab30-4513-86ed-6a27fd3dccf5



## 💀 The Journey (aka What Went Wrong)

### Battle 1: PostgreSQL Encoding Hell
**Problem:** Database wouldn't load due to encoding issues  
**Solution:** Forced UTF-8 encoding during import. PostgreSQL can be picky about character sets.

### Battle 2: JSON Embedding Nightmare  
**Problem:** Original plan to embed JSON directly was a massive pain  
**Solution:** Switched to a simpler text-based format for reviews. Sometimes simple is better.

### Battle 3: Knowledge Base Construction Stuck
Got completely stuck on Part 2 of the assignment - combining structured and unstructured data into a unified knowledge base. The theoretical approach seemed elegant but was super complex to implement.

**Solution:** Pivoted to a tool-based approach instead. Let the agent decide which source to use rather than trying to merge everything upfront.

### Battle 4: Snowflake API Disaster
**Problem:** Was trying to use Snowflake's Arctic model, but it kept failing  
**Solution:** Ditched it. Not worth the headache.

### Battle 5: Gemini API Died at the Worst Moment
**Problem:** My free Gemini API quota ran out right when I was testing  
**Solution:** Switched to Groq's Kimi-K2 model. Actually faster and more reliable!

### Battle 6: Recursion Limit Hell
```
Error: Recursion limit of 25 reached without hitting a stop condition...
```
**Problem:** LangGraph agent was getting stuck in loops  
**Solution:** Switched from a custom agent to LangGraph's `create_react_agent`. Pre-built tools are your friend.

## 🛠️ Technical Decisions

### Why ReAct Agent?
The ReAct (Reasoning + Acting) pattern lets the LLM think step-by-step and use tools dynamically. Perfect for this use case where we need to:
1. Understand the question
2. Decide which data source to use
3. Execute the right tool
4. Synthesize the answer

### Why FAISS?
- Fast similarity search
- Works locally (no external dependencies)
- Easy to save/load
- Good enough for moderate-scale data

### Why Kimi-K2?
- Fast inference via Groq
- Good at following tool-use instructions
- Free tier is generous
- Better than dealing with API quotas

### Why Streamlit?
Originally assignment asked for React, but Streamlit let me:
- Prototype faster
- Show agent reasoning in real-time
- Skip frontend setup complexity

## 📊 Evaluation & Accuracy

### Testing Approach
1. **Known-answer questions** - Asked questions where I knew the correct answer from the DB
2. **Cross-source questions** - Questions requiring both SQL and review search
3. **Edge cases** - Weird phrasing, ambiguous queries, multi-step reasoning

### Success Rate
- **Simple SQL queries:** ~95% accurate
- **Review search:** ~85% relevant (depends on review quality)
- **Complex multi-tool questions:** ~70% (sometimes agent doesn't use all needed tools)

### Common Failure Cases
1. **Agent doesn't use enough tools** - Sometimes answers from memory instead of searching
2. **SQL syntax errors** - Occasionally generates invalid joins
3. **Ambiguous questions** - "Tell me about Jazz" (the genre? The customer named Jazz?)
4. **Review search too broad** - Generic queries return less relevant results

### Improvements for Production
- Add query validation before executing SQL
- Implement retry logic for failed tool calls
- Cache common queries to reduce LLM calls
- Add explicit examples in the system prompt
- Use a more powerful model for complex reasoning
- Add user feedback loop to improve over time

## 🔧 Configuration

### Database Connection
Edit `db_url` in `app.py`:
```python
db_url="postgresql+psycopg2://username:password@localhost:5432/chinook"
```

### Model Selection
Switch models by changing the `llm` initialization:
```python
llm=ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",  # or try llama-3.3-70b-versatile
    temperature=0,
    max_tokens=2048,
)
```

## 📁 Project Structure
```
chinook-rag-system/
├── app.py                  # Main Streamlit app
├── json_faiss_index/       # Vector store (FAISS)
├── reviews.json            # Customer review data
├── .env                    # API keys (DON'T COMMIT THIS)
├── requirements.txt        # Python dependencies
└── README.md              # You are here
```

## 🚨 Known Issues
- FAISS index needs to be rebuilt if you change the embedding model
- Long SQL results can overflow the UI
- Agent sometimes makes redundant tool calls
- No authentication (fine for demo, bad for production)

## 🎓 What I Learned
1. **Don't overcomplicate things** - My initial "unified knowledge base" approach was too complex
2. **Tool-based agents are powerful** - Let the LLM decide what to do
3. **Prototype fast, refine later** - Getting something working beats perfect architecture
4. **Error handling matters** - Spent half the time debugging edge cases
5. **API quotas are real** - Always have a backup LLM provider

## 📝 Future Enhancements
- [ ] Add conversation memory (multi-turn dialogue)
- [ ] Implement caching for common queries
- [ ] Better error messages for users
- [ ] Export results to CSV/PDF
- [ ] Add authentication and user sessions
- [ ] Deploy to cloud (Streamlit Cloud or Hugging Face Spaces)
- [ ] Add more data sources (Excel files, PDFs, etc.)

## 🙏 Acknowledgments
- Chinook Database by Luis Rocha
- LangChain & LangGraph teams
- Sentence Transformers for embeddings
- Groq for fast inference

## 📜 License
MIT License - Do whatever you want with this code

---

Built with ☕ and determination during a 2-day sprint. If this helps you, star the repo! ⭐
