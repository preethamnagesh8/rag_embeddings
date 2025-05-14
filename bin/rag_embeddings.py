from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage

from core.custom_embeddings import NewEmbeddings
from core.custom_chat_completions import NewGPT
from dotenv import load_dotenv
from lib.rag_chunking import chunk_text
from lib.retrieve_in_memory import retrieve_similar_docs
from lib.store_in_memory import create_vectorstore
import os

# Load .env from custom path
load_dotenv(dotenv_path="config/.env")

# Use with your embedding class
embeddings = NewEmbeddings(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_RAG_API_SECRET"),
    model=os.getenv("EMBEDDING_MODEL"),
)

llm = NewGPT(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_RAG_API_SECRET"),
    model=os.getenv("OPENAI_CHAT_MODEL")
)

if __name__ == "__main__":
    long_text = """
    Shruthi had always been drawn to the unknown. As a young girl in the small coastal town of Miraval, she spent her afternoons charting imaginary constellations and recording mysterious signals on her homemade radio. Her room, cluttered with old telescopes, mechanical scraps, and notebooks full of codes, was less a bedroom and more a command center for her endless curiosity. While her friends played games outside, Shruthi decoded whispers from distant frequencies, convinced the universe was sending her puzzles only she could solve. By the time she was twenty-three, Shruthi had turned her fascination into a career. Working as an experimental systems engineer at a private aerospace lab, she discovered an anomalous data packet during a routine satellite calibration. Unlike anything she’d seen before, the signal pulsed with a rhythm that mirrored Morse code—but in a sequence too complex to be manmade. She spent weeks decoding it, and when she finally succeeded, she uncovered a map—a precise set of astronomical coordinates pointing to a region of space long thought to be void. On a quiet winter night, Shruthi launched a small probe of her own design into the coordinates. Weeks passed with no transmission, until one morning her receiver crackled to life with a signal that wasn’t just a response—it was a message. A voice, synthetic but oddly human, thanked her for listening. That day, Shruthi became more than a scientist. She became the first Earthling to open a conversation across the stars—not because she searched the loudest, but because she listened the longest.
    """

    # 1. Chunk
    chunks = chunk_text(long_text)

    # 2. Embed + Store
    vectorstore = create_vectorstore(chunks, embeddings)

    # 3. Query
    query = "What was the name of the coastal town where Shruthi grew up?"
    results = retrieve_similar_docs(vectorstore, query)

    print("\nTop matching chunks:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")

    context = "\n\n".join([doc.page_content for doc in results])

    messages = [
        SystemMessage(content="You are a helpful assistant. Use the context to answer the question."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    response = llm.invoke(messages)
    print("\nAnswer:\n", response.content)
