import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

os.environ["ACTIVELOOP_TOKEN"] = ("eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"
                                  ".eyJpZCI6Im1yaW8iLCJhcGlfa2V5IjoiMi1INHBkZnY2M0syS0Rtc2JhVjNLSnZ2NUlwWTBjenJiaTFhZkxOZzlMYkZLIn0.")
os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDrAvNPH3FTcael_LnHagXYybNu1--IVLc"
os.environ["GOOGLE_CSE_ID"] = "974b03204274e4f99"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    summarization_template = "Summarize the following text in less than ten words: {text}"
    summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
    summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
    text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."
    summarized_text = summarization_chain.predict(text=text)
    print(summarized_text)
