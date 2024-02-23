from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

if __name__ == '__main__':
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Load the summarization chain
    summarize_chain = load_summarize_chain(llm)

    # TODO: Load PDF file from message.
    # TODO: Save file into a database.
    # TODO: Add automatically some key words for later search, a title and a summary for the users.
    # TODO: Add the content to a DeepLake database.

    # Load the document using PyPDFLoader
    document_loader = PyPDFLoader(file_path="/home/mrio/itg-rag-practice/lang-chain-practice/example-pdf.pdf")
    document = document_loader.load()

    # Summarize the document
    summary = summarize_chain(document)
    print(summary['output_text'])