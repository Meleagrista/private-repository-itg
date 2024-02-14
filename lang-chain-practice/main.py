import os

from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

os.environ["ACTIVELOOP_TOKEN"] = ("eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"
                                  ".eyJpZCI6Im1yaW8iLCJhcGlfa2V5IjoiMi1INHBkZnY2M0syS0Rtc2JhVjNLSnZ2NUlwWTBjenJiaTFhZkxOZzlMYkZLIn0.")
os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDrAvNPH3FTcael_LnHagXYybNu1--IVLc"
os.environ["GOOGLE_CSE_ID"] = "974b03204274e4f99"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    search = GoogleSearchAPIWrapper()

    tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.run,
    )

    print(tool.run("Obama's first name?"))

