# import json
from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import TextLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import requests
from newspaper import Article
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# Load environment variables from .env file
load_dotenv()


def article_summarizer():

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
    session = requests.Session()
    try:
        response = session.get(article_url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(article_url)
            article.download()
            article.parse()

            # prepare template for prompt
            template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

            Here's the article you need to summarize.

            ==================
            Title: {article_title}

            {article_text}
            ==================

            Now, provide a summarized version of the article in a bulleted list format.
            """

            # format prompt
            prompt = template.format(article_title=article.title, article_text=article.text)
            # generate summary
            summary = chat([HumanMessage(content=prompt)])
            print(summary.content)
        else:
            print(f"Failed to fetch article at {article_url}")
            
    except Exception as e:
        print(f"Error occurred while fetching article at {article_url}: {e}")


def gpt_all():
    template = """Question: {question}

        Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = GPT4All(model="./models/ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What happens when it rains somewhere?"
    llm_chain.run(question)

def document_loader():
    # text to write to a local file
    # taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
    text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
    Google is offering developers access to one of its most advanced AI language models: PaLM.
    The search giant is launching an API for PaLM alongside a number of AI enterprise tools
    it says will help businesses “generate text, images, code, videos, audio, and more from
    simple natural language prompts.”

    PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
    Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
    PaLM is a flexible system that can potentially carry out all sorts of text generation and
    editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
    example, or you could use it for tasks like summarizing text or even writing code.
    (It’s similar to features Google also announced today for its Workspace apps like Google
    Docs and Gmail.)
    """

    # write text to local file
    with open("my_file.txt", "w") as file:
        file.write(text)

    # use TextLoader to load text from local file
    loader = TextLoader("my_file.txt")
    docs_from_file = loader.load()

    # create a text splitter
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # split documents into chunks
    docs = text_splitter.split_documents(docs_from_file)

    # print(len(docs))

    # Before executing the following code, make sure to have
    # your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Before executing the following code, make sure to have your
    # Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

    # create Deep Lake dataset
    # TODO: use your organization id here. (by default, org id is your username)
    my_activeloop_org_id = "mrio"
    my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

    # add documents to our Deep Lake dataset
    db.add_documents(docs)

    # create retriever from db
    retriever = db.as_retriever()

    # create a retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-3.5-turbo-instruct"),
        chain_type="stuff",
        retriever=retriever
    )

    query = "How Google plans to challenge OpenAI?"
    # response = qa_chain.run(query)
    # print(response)

    # create GPT3 wrapper
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    # create compressor for the retriever
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    # retrieving compressed documents
    retrieved_docs = compression_retriever.get_relevant_documents(
        "How Google plans to challenge OpenAI?"
    )
    print(retrieved_docs[0].page_content)

def other_loader():
    loader = TextLoader('file_path.txt')
    documents = loader.load()

    """from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
    pages = loader.load_and_split()
    
    from langchain.document_loaders import SeleniumURLLoader

    urls = [
        "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",
        "https://www.youtube.com/watch?v=6Zv6A_9urh4&t=112s"
    ]
    
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()

    """


if __name__ == "__main__":
    document_loader()
