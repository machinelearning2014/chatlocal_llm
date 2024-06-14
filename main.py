import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def get_conversation_history(messages):
    history = ""
    for message in messages[:-1]:  # Exclude the last message (the current user input)
        if message["role"] == "user":
            history += "Human: " + message["content"] + "\n"
        else:
            history += "Assistant: " + message["content"] + "\n"
    return history

def create_chain(system_prompt):
    stream_handler = StreamHandler(st.empty())
    callback_manager = CallbackManager([stream_handler])

    model_file = "mistral-7b-instruct-v0.2.Q2_K.gguf"
    #model_file = "/home/letro/"

    #selected_model = st.selectbox("Select a model", models)
    #model_file += selected_model

    llm = LlamaCpp(
        model_path=model_file,
        temperature=0,
        max_tokens=512,
        top_p=1,
        callback_manager=callback_manager,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=4096,
        stop=["[INST]"],
        verbose=False,
        streaming=True,
    )

    template = """
    <s>[INST]{}[/INST]</s>

    [INST]Conversation History:\n{}

    Human: {}[/INST]
    """.format(system_prompt, "{history}", "{question}")

    prompt = PromptTemplate(
        template=template,
        input_variables=["history", "question"]
    )

    llm_chain = prompt | llm

    return llm_chain

st.set_page_config(
    page_title="ChatLocalLLM"
)

st.header("ChatLocalLLM")

system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt")

llm_chain = create_chain(system_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Your message here", key="user_input"):
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    conversation_history = get_conversation_history(st.session_state.messages)
    response = llm_chain.invoke({"history": conversation_history, "question": user_prompt})

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
