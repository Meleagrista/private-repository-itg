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
    # initialize Hub LLM
    hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
        model_kwargs={'temperature': 0}
    )

    multi_template = """Answer the following questions one at a time.

    Questions:
    {questions}

    Answers:
    
    """
    long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

    llm_chain = LLMChain(
        prompt=long_prompt,
        llm=hub_llm
    )

    qs_str = (
            "1. What is the capital city of France?\n" +
            "2. What is the largest mammal on Earth?\n" +
            "3. Which gas is most abundant in Earth's atmosphere?\n" +
            "4.What color is a ripe banana?\n"
    )
    res = llm_chain.run(qs_str)
    print(res)
