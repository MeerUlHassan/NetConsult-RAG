import streamlit as st
from copymain import app, HumanMessage

st.set_page_config(page_title="NetConsult Virtual Assistant", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    .user-msg {
        background: #e0f7fa;
        text-align: right;
    }
    .bot-msg {
        background: #f1f8e9;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 NetConsult Support Assistant")
st.caption("Your AI-powered business growth partner.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def get_response(user_input):
    # Use the same workflow as your CLI app
    st.session_state["messages"].append(HumanMessage(user_input))
    config = {"configurable": {"thread_id": "office"}}
    output = app.invoke({"messages": st.session_state["messages"]}, config)
    st.session_state["messages"].append(output["messages"][-1])
    return output["messages"][-1].content

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", key="user_input", placeholder="Ask me anything about NetConsult...")
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        response = get_response(user_input)

# Display chat history
for msg in st.session_state["messages"]:
    if msg.type == "human":
        st.markdown(f'<div class="stChatMessage user-msg">🧑‍💼 <b>You:</b> {msg.content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="stChatMessage bot-msg">🤖 <b>Assistant:</b> {msg.content}</div>', unsafe_allow_html=True)