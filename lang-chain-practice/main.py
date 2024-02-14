import os

from langchain import HuggingFaceHub, LLMChain
from langchain import PromptTemplate

os.environ["ACTIVELOOP_TOKEN"] = ("eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"
                                  ".eyJpZCI6Im1yaW8iLCJhcGlfa2V5IjoiMi1INHBkZnY2M0syS0Rtc2JhVjNLSnZ2NUlwWTBjenJiaTFhZkxOZzlMYkZLIn0.")
os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDrAvNPH3FTcael_LnHagXYybNu1--IVLc"
os.environ["GOOGLE_CSE_ID"] = "974b03204274e4f99"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(
        template=template,
        input_variables=['question']
    )

    # user question
    question = "What is the capital city of France?"

    # initialize Hub LLM
    hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
        model_kwargs={'temperature': 0}
    )

    # create prompt template > LLM chain
    llm_chain = LLMChain(
        prompt=prompt,
        llm=hub_llm
    )

    # ask the user question about the capital of France
    print(llm_chain.run(question))
