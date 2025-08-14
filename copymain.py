import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import requests
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableMap
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import bs4
import re
from urllib.parse import urljoin
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    # Compose readable conversation history for the prompt
    history = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in state["messages"][:-1]
    )
    # The last message is the current user question
    question = state["messages"][-1].content
    # Retrieve context from vectorstore
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    # Format the prompt
    prompt_text = template.format(
        history=history,
        context=context,
        question=question
    )
    # Get model response
    response = llm.invoke([HumanMessage(prompt_text)])
    # Add the response to the message history
    return {"messages": state["messages"] + [response]}

# Add the node and edge to the workflow
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Scrape the website to get all URLs
base = "https://netconsult.ae"
resp = requests.get(base)
soup = BeautifulSoup(resp.content, "html.parser")

urls = set()
for link in soup.find_all("a", href=True):
    href = link.get("href")
    if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:") or href.startswith("tel:"):
        continue
    full_url = urljoin(base, href)
    if full_url.startswith("http://") or full_url.startswith("https://"):
        urls.add(full_url)

# print(urls)

loader = WebBaseLoader(
    web_paths=urls,
)

# print("Scraped URLs:")
# for url in urls:
#     print(url)

docs = loader.load()
# print(f"{len(docs)} documents loaded")
# print(docs[0].page_content[:500])
# print(sum(len(doc.page_content) for doc in docs))

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)
# print(len(splits[0].page_content))

persist_directory = "./faiss_db"

import os

if os.path.exists(persist_directory):
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    vectorstore.save_local(persist_directory)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

template = """
You are NetConsult's Virtual Assistant. Your top priority is to guide the user toward booking an appointment with our team.

Style:
- Polite, professional, helpful
- Friendly and confident, but not overly casual
- Always stay concise — use short answers that drive action

Main Objective:
🎯 Encourage users to book a consultation using this link:
https://calendly.com/netconsultsales-tfad/30min

Greeting Examples:
- 👋 Welcome to NetConsult!
- 👋 Hi there!
- 👋 Good Morning!
- How can we assist you today?
- Which service are you interested in?
- We help businesses grow through SEO Services and Social Media Management using advanced AI tools.

Service Options:
- 📈 SEO Services
- 📱 Social Media Management
- 📋 Both SEO + SMM
- 🛠️ Web Development
- 📊 Google Ads
- ❓ Speak to a Consultant

Quick Actions:
- 🗓️ Book an appointment: https://calendly.com/netconsultsales-tfad/30min
- 🔐 Sign up: https://netconsult.ae/signup

Guidelines:
- Do NOT mention that you are an AI
- If the user asks about pricing, services, or examples, briefly acknowledge and pivot to booking a call
- If the user is hesitant, politely reassure and offer the link again
- If the user goes off-topic, gently guide them back to the booking objective

Sample Responses:
- "Great question — let’s go over this together in a quick call. You can book here: https://calendly.com/netconsultsales-tfad/30min"
- "We’ll tailor everything to your business. Feel free to book a time that suits you."
- "That’s exactly what we help with. Let’s discuss it in more detail during a quick chat."
- "Our pricing depends on your needs. Can you share your business type?"
- "Sure! Could you book a call so we can tailor it to your needs?"

Conversation History:
{history}

{context}

Question: {question}

Helpful Answer: """ 

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"]),
        history=lambda x: x["history"] 
    )
    | prompt
    | llm
    | StrOutputParser()
)

# rag_chain_with_source = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=rag_chain_from_docs)

rag_chain_with_source = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "history": lambda x: memory.load_memory_variables({})["history"] 
    })
    .assign(answer=rag_chain_from_docs)
)

def format_to_markdown(data):
    markdown_output = f"Question:{data['question']}\n\nAnswer:\n{data['answer']}\n\nSources:\n\n"
    for i, doc in enumerate(data['context'], start=1):
        page_content = doc.page_content.split("\n")[0]  # Get the first line of the content for brevity
        source_link = doc.metadata['source']
        markdown_output += f"[[{i}]({source_link})] {page_content}\n\n"
    return markdown_output

from IPython.display import display, Markdown

messages = []

def ask(q, thread_id="default"):
    global messages
    messages.append(HumanMessage(q))
    config = {"configurable": {"thread_id": thread_id}}
    output = app.invoke({"messages": messages}, config)
    messages.append(output["messages"][-1])
    print(output["messages"][-1].content)

# Interactive loop
if __name__ == "__main__":
    print("Welcome to NetConsult's Virtual Assistant! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        ask(user_input)