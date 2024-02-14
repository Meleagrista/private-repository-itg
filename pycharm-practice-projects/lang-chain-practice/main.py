from langchain_openai import OpenAI

API_KEY = "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"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Before executing the following code, make sure to have
    # your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, openai_api_key=API_KEY)

    text = ("Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and "
            "prefers outdoor activities.")
    print(llm.invoke(text))
