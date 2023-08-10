from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
import nest_asyncio

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# product urls to scrape
mattress_urls = ['https://www.saatva.com/mattresses/saatva-classic',
 'https://www.saatva.com/mattresses/loom-and-leaf',
 'https://www.saatva.com/mattresses/saatva-latex-hybrid',
 'https://www.saatva.com/mattresses/solaire',
 'https://www.saatva.com/mattresses/memory-foam-hybrid',
 'https://www.saatva.com/mattresses/zenhaven',
 'https://www.saatva.com/mattresses/saatva-hd',
 'https://www.saatva.com/mattresses/saatva-rx',
 'https://www.saatva.com/mattresses/saatva-youth',
 'https://www.saatva.com/mattresses/crib-mattress',
 'https://www.saatva.com/mattresses/dog-bed']

#####
# location to store vector db
db_path  = 'vectorstores/db_faiss'

#@root_validator(pre=False, skip_on_failure=True)
def create_vector_db():
  # scrape product pages
  #nest_asyncio.apply()
  loader = UnstructuredURLLoader(mattress_urls)
  data = loader.load()
  text_splitter = CharacterTextSplitter(separator='\n', chunk_size = 1100, chunk_overlap = 50)
  chunks = text_splitter.split_documents(data)
  
  embeddings = HuggingFaceEmbeddings()

  db = FAISS.from_documents(chunks, embeddings)
  db.save_local(db_path)

if __name__ == '__main__':
  create_vector_db()

