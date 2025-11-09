import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import re

# Load up those environment variables
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

st.title("Chinook RAG Q&A System (ReAct)")

# Setting up our LLM - using Groq's Kimi model
llm=ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0, 
)

# Database connection PostgreSQL 
db_url="postgresql+psycopg2://postgres:19283746@localhost:5432/chinook"
try:
    db=SQLDatabase.from_uri(db_url)
except Exception as e:
    st.error(f"DB Connection Failed: {e}")
    st.stop()

def load_vector_store():
    try:
        embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  
        )
        vector_store=FAISS.load_local(
            "json_faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True  
        )
        return vector_store
    except Exception as e:
        st.error(f"FAISS Load Error: {e}")
        st.stop()

vector_store=load_vector_store()

# Tool 1: Execute SQL queries against Chinook database
@tool
def sql_query(query: str) -> str:
    """Execute a SQL query on the Chinook PostgreSQL database.
    Use for: customer data, sales, invoices, albums, artists, tracks, totals, counts.
    """
    try:
        # Clean up the query
        clean_query=re.sub(r'```sql|```', '', query).strip().rstrip(';')
        result=db.run(clean_query)
        return f"Results:\n{result}"
    except Exception as e:
        return f"SQL Error: {str(e)}"

# Tool 2: Get schema info for the agent to understand table structure
@tool
def get_schema(table_name: str="") -> str:
    """Get database schema information.
    
    Args:
        table_name: Specific table name, or leave empty to see all tables.
    """
    try:
        if table_name:
            return db.get_table_info([table_name])
        else:
            return """Tables: album, artist, customer, employee, invoice, invoice_line, track, genre, media_type, playlist
            
Relationships:
- customer.support_rep_id → employee.employee_id
- invoice.customer_id → customer.customer_id
- invoice_line.invoice_id → invoice.invoice_id"""
    except Exception as e:
        return f"Error: {str(e)}"

# Tool 3: Search through customer reviews using semantic similarity
@tool
def search_reviews(query: str) -> str:
    """Search customer reviews and album reviews using semantic search.
    
    Use for: opinions, complaints, feedback, ratings, customer comments.
    """
    try:
        # Grab top 3 most relevant reviews
        docs=vector_store.similarity_search(query, k=3)
        if not docs:
            return "No relevant reviews found."
        
        results=[]
        for i, doc in enumerate(docs, 1):
            results.append(f"Review {i}:\n{doc.page_content}\n")
        return "\n".join(results)
    except Exception as e:
        return f"Error: {str(e)}"

# Package up our tools for the agent
tools=[sql_query, get_schema, search_reviews]

# Keep the system prompt simple and clear - agents work better this way
system_prompt="""You are a helpful assistant for the Chinook music store database.

Use the available tools to answer questions:
- search_reviews: for customer opinions/complaints/feedback
- sql_query: for factual data (sales, customers, invoices)
- get_schema: to understand database structure

For complex questions, use multiple tools in sequence. Always provide a complete answer."""

# Create our ReAct agent - this baby will think step by step
agent=create_react_agent(llm, tools, prompt=system_prompt)

# --- Streamlit UI stuff ---
st.write("Ask anything about the Chinook database or customer reviews")

query=st.text_input("Your question:")

if st.button("Ask") or query:
    if query:
        st.write("---")
        
        with st.container():
            st.write("**Agent Working:**")
            
            step=0
            final_answer=None
            
            try:
                # Stream the agent's thinking process so we can watch it work
                for event in agent.stream(
                    {"messages": [HumanMessage(content=query)]},
                    stream_mode="values"
                ):
                    messages=event.get("messages", [])
                    if messages:
                        last=messages[-1]
                        
                        # Agent is calling a tool - let's show which one
                        if hasattr(last, "tool_calls") and last.tool_calls:
                            step+=1
                            st.write(f"**Step {step}:**")
                            for tc in last.tool_calls:
                                st.write(f" `{tc['name']}`")
                                # Show args but truncate if too long
                                st.write(f" `{str(tc['args'])[:80]}...`")
                        
                        # Got a result back from a tool
                        elif hasattr(last, "content") and hasattr(last, "name"):
                            with st.expander("📊 Result", expanded=False):
                                # Truncate long results so UI doesn't explode
                                st.text(last.content[:400])
                        
                        # This should be the final answer from the agent
                        elif hasattr(last, "content") and last.content:
                            if not hasattr(last, "tool_calls"):
                                final_answer=last.content
                
                # Show the final answer in a nice format
                st.write("---")
                st.write("**Answer:**")
                if final_answer:
                    st.success(final_answer)
                else:
                    st.info("Done")
                    
            except Exception as e:
                st.error(f"Error: {e}")