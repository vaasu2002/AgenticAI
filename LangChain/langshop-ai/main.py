import os
import logging
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductInfo(BaseModel):
    product_name: str = Field(..., description="Name of the product")
    product_details: str = Field(..., description="Brief description of the product")
    price_inr: str = Field(..., description="Price with ₹ symbol, e.g., ₹79999")

parser = JsonOutputParser(pydantic_object=ProductInfo)

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful product assistant for the Indian Market. "
            "Return output in JSON format with fields: 'product_name', 'product_details', and 'price_inr'. "
            "Include only relevant details. No extra commentary."
        ),
        ("user", "{input}"),
        ("assistant", "{format_instructions}")
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

if __name__ == "__main__":
    try:
        product_query = "Tell me about iPhone 15."
        response: ProductInfo = chain.invoke({"input": product_query})
        print(response.json(indent=2))
    except Exception as e:
        logger.error("Error while fetching product info", exc_info=True)
