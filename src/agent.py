from src.agent import build_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langhchain.memory import ConversationBufferMemory

agent = build_agent('C:/Users/husain_althagafi/work/storage/Qwen2.5-7B-Instruct')

prompt_template = PromptTemplate.from_template(
    input_variables=['context', 'question'],
    template="You are a helpful assistant. Here is the context: {context}. Please answer the following question: {question}"
)

basic_chain = LLMChain(
    llm=agent,
    prompt=prompt_template,
    verbose=True
)

response = basic_chain.run(
    context="My name is Husain.",
    question="What is my name?"
)

print(f'Response: {response}') 
