import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

"""from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate"""

"""from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI"""

"""# Import necessary modules
from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader"""

"""# Import necessary modules
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader"""

"""from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)"""

"""from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate"""

"""from transformers import AutoTokenizer"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # chat = ChatOpenAI(model_name="gpt-4", temperature=0)

    """summarization_template = "Summarize the following text in less than ten words: {text}"
    summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
    summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
    translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
    translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"], template=translation_template)
    translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
    text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."
    summarized_text = summarization_chain.predict(text=text)
    source_language = "English"
    target_language = "French"
    translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)
    print(summarized_text)
    print(translated_text)"""

    """# Download and load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # print(tokenizer.vocab)
    token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")
    print("Tokens:   ", tokenizer.convert_ids_to_tokens(token_ids))
    print("Token IDs:", token_ids)"""

    """template = "You are an assistant that helps users find information about movies."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Find information about the movie {movie_title}."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    response = chat(chat_prompt.format_prompt(movie_title="Inception").to_messages())
    print(response.content)"""

    """# Load the summarization chain
    summarize_chain = load_summarize_chain(llm)
    # Load the document using PyPDFLoader
    document_loader = PyPDFLoader(file_path="/home/mrio/itg-rag-practice/lang-chain-practice/example-pdf.pdf")
    document = document_loader.load()
    # Summarize the document
    summary = summarize_chain(document)
    print(summary['output_text'])"""

    """prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run("what is the meaning of life?"))"""

    """prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}? Return only three names.",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run("wireless headphones"))"""

    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="Translate the following sentence: I love programming.")
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates French to English."),
            HumanMessage(content="Translate the following sentence: J'aime la programmation.")
        ],
    ]
    print(chat.generate(batch_messages))

