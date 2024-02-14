import os

from langchain_openai import OpenAI
from langchain.callbacks import get_openai_callback

os.environ["ACTIVELOOP_TOKEN"] = ("eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"
                                  ".eyJpZCI6Im1yaW8iLCJhcGlfa2V5IjoiMi1INHBkZnY2M0syS0Rtc2JhVjNLSnZ2NUlwWTBjenJiaTFhZkxOZzlMYkZLIn0.")
os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDrAvNPH3FTcael_LnHagXYybNu1--IVLc"
os.environ["GOOGLE_CSE_ID"] = "974b03204274e4f99"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)

    with get_openai_callback() as cb:
        result = llm.invoke("Tell me a joke")  # Using invoke instead of __call__
        print(cb)
