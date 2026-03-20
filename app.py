import os
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 🛡️ 屏蔽代理

# 🔑 你的 API Keys
# 让代码去 secrets 保险箱里拿钥匙
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# 页面设置
st.set_page_config(page_title="AI 旅游管家", page_icon="✈️")
# ✨ 新的 UI 优化代码
st.markdown("""
    <style>
    /* 隐藏顶部的 Streamlit 菜单栏，看起来更像原生 App */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    /* 缩小手机端的顶部留白 */
    .block-container {padding-top: 2rem;}
    </style>
    
    <h3 style='text-align: center; margin-bottom: 5px;'>✈️ 你的专属 AI 旅游管家</h3>
    <p style='text-align: center; color: gray; font-size: 0.85rem;'>
        基于 LangGraph + 智谱大模型构建<br>支持全网搜索与长效记忆
    </p>
    """, unsafe_allow_html=True)

# 初始化 Agent 和记忆
if "agent_executor" not in st.session_state:
    search_tool = TavilySearchResults(max_results=3)
    llm = ChatOpenAI(
        model="glm-4-flash",             
        api_key=st.secrets["ZHIPU_API_KEY"], # 👈 这里改成从保险箱读取
        base_url="https://open.bigmodel.cn/api/paas/v4/", 
        temperature=0.7
    )
    memory = MemorySaver()
    st.session_state.agent_executor = create_react_agent(llm, [search_tool], checkpointer=memory)

    # 换个新包厢号，清空刚才的报错记忆
    st.session_state.config = {"configurable": {"thread_id": "web_trip_002"}}

    # 💡 修复点：记录是否为第一轮对话
    st.session_state.is_first_turn = True

    # 初始化网页聊天记录
    st.session_state.chat_history = [
        {"role": "assistant", "content": "你好！我是你的专属旅游管家。告诉我你们想去哪里玩？有什么忌口或者偏好吗？"}]

# 渲染历史聊天记录
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理用户输入
user_input = st.chat_input("输入你的旅游想法...")

if user_input:
    # 1. 把用户的话显示在界面上
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 2. 调用 Agent 思考并生成回复
    with st.spinner("🤖 管家正在全网搜索并规划中，请稍候..."):

        # 💡 核心修复逻辑：第一轮对话打包发送系统规则
        if st.session_state.is_first_turn:
            system_prompt = "你是一个贴心的旅游规划师。请调用搜索工具获取最新信息，并严格牢记用户提出的偏好和忌口。"
            messages_to_send = [("system", system_prompt), ("user", user_input)]
            st.session_state.is_first_turn = False
        else:
            messages_to_send = [("user", user_input)]

        result = st.session_state.agent_executor.invoke(
            {"messages": messages_to_send},
            st.session_state.config
        )
        ai_response = result["messages"][-1].content

    # 3. 把 AI 的回复显示在界面上
    st.chat_message("assistant").write(ai_response)
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
