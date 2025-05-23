""" Populate chroma database"""
import os
from dotenv import load_dotenv
import yaml
import csv
import logging
from langchain_openai import OpenAIEmbeddings
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional

#load env variables
load_dotenv(override=True)

#load YAML variables
#read YAMl

with open('config_parameters.yml', 'r') as f:
    data = yaml.safe_load(f)

VECTOR_DB_PATH = data["chroma"]["VECTOR_DB_PATH"]
SP500_INFO_PATH = data["dabatase"]["SP500_INFO_PATH"]
MODEL_EMBEDDINGS = data["embeddings"]["MODEL_EMBEDDINGS"]
COLLECTION_NAME = data["chroma"]["COLLECTION_NAME"]



class ChromaEmbeddingProcessor:
    """
    Clase para procesar archivos CSV, generar embeddings con OpenAI y almacenar en ChromaDB
    """
    
    def __init__(self):
        """
        Inicializa el procesador
        
        """
        self.VECTOR_DB_PATH = VECTOR_DB_PATH
        self.SP500_INFO_PATH = SP500_INFO_PATH
        self.MODEL_EMBEDDINGS = MODEL_EMBEDDINGS
        self.COLLECTION_NAME = COLLECTION_NAME

        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        
        # Inicializar cliente OpenAI
        self.openaiembeddings_model_client = self._init_openai_client()
        
        # Inicializar cliente ChromaDB
        self.chroma_client = self._init_chroma_client()
        
        # Variables para almacenar datos
        self.file_lines = []
        self.list_text = []
        self.embeddings = []
        self.chroma_collection = None
        
    def _init_openai_client(self) -> OpenAIEmbeddings:
        """Inicializa el cliente de OpenAI"""
        return OpenAIEmbeddings(model=self.MODEL_EMBEDDINGS)

    
    def _init_chroma_client(self):
        """Inicializa el cliente de ChromaDB"""
        # Crear directorio si no existe
        Path(self.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

        return chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    def read_csv(self) -> None:
        """
        Lee un archivo CSV y guarda su contenido
        
        """
        try:
            with open(self.SP500_INFO_PATH, mode ='r', encoding='utf-8-sig') as file:    
                csvFile = csv.DictReader(file)
                for lines in csvFile:
                        self.file_lines.append(lines)
        
        except Exception as e:
            self.logger.error(f"Error al leer CSV: {e}")
            raise
    
    def create_custom_text_list(self) -> None:
        """
        Crea una lista de texto personalizado basado en las columnas del CSV
        
        """
        if self.file_lines is None:
            raise ValueError("Primero debe leer un archivo CSV")
        
        for i in self.file_lines:
            text_input = f"""ticker: {i['ticker']} shortName:  {i['shortName']} country:  {i['country']} 
            industry:  {i['industry']} sector:  {i['sector']} fullTimeEmployees:  {i['fullTimeEmployees']}
 {i['companyOfficers_title']}: {i['companyOfficers_name']} longBusinessSummary:  {i['longBusinessSummary']}"""
    
            self.list_text.append(text_input)
        
        self.logger.info(f"Lista de texto creada: {len(self.list_text)} elementos")
    
    def generate_embeddings(self) -> None:
        """
        Genera embeddings para cada texto usando OpenAI
        
        """
        if not self.list_text:
            raise ValueError("Primero debe crear la lista de textos")
        

        try:
            self.embeddings = self.openaiembeddings_model_client.embed_documents(self.list_text)
        except Exception as e:
            self.logger.error(f"Error al generar embeddings: {e}")
            raise

    def setup_chroma_collection(self, reset_collection: bool = True) -> chromadb.Collection:
        """
        Configura la colección de ChromaDB
       
        """
        try:
            # Verificar si la colección existe
            existing_collection_names = [collection for collection in self.chroma_client.list_collections()]
            
            if self.COLLECTION_NAME in existing_collection_names:
                if reset_collection:
                    self.logger.info(f"Borrando colección existente: {self.COLLECTION_NAME}")
                    self.chroma_client.delete_collection(name=self.COLLECTION_NAME)
                else:
                    self.logger.info(f"Usando colección existente: {self.COLLECTION_NAME}")
                    return self.chroma_client.get_collection(name=self.COLLECTION_NAME)
            
            # Crear nueva colección
            self.logger.info(f"Creando nueva colección: {self.COLLECTION_NAME}")
            collection = self.chroma_client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Embeddings from CSV data"}
            )
            
            return collection
            
        except Exception as e:
            self.logger.error(f"Error al configurar colección ChromaDB: {e}")
            raise
    
    def populate_chroma_db(self, 
                          metadata_columns: List[str] = None,
                          batch_size: int = 100) -> None:
        """
        Llena la base de datos ChromaDB con textos y embeddings
        
    
        """
        if not self.embeddings or not self.list_text:
            raise ValueError("Debe generar embeddings antes de llenar ChromaDB")
        
        self.chroma_collection = self.setup_chroma_collection()
        
        # Preparar metadata
        try:
            for i, file in enumerate(self.file_lines):
                document = self.list_text[i]
                vector = self.embeddings[i]
                metadata = [{"ticker": file['ticker'],
                            "shortName":  file['shortName'],
                            "country":  file['country'],
                            "industry":  file['industry'],
                            "sector":  file['sector'],
                            "officer_name": file['companyOfficers_name']}]
                self.chroma_collection.add(
                    ids=str(i),
                    documents=document,
                    embeddings=vector,
                    metadatas=metadata)

            self.logger.info(f"Insertados {i} documentos en ChromaDB")
                
        except Exception as e:
            self.logger.error(f"Error al insertar lote en ChromaDB: {e}")
            raise
        
        self.logger.info("Base de datos ChromaDB poblada exitosamente")
    
    def run(self) -> None:
        """
        Run all

        """
        self.logger.info("Iniciando procesamiento completo...")
        
        # 1. Leer CSV
        self.read_csv()
        
        # 2. Crear lista de textos
        self.create_custom_text_list()
        
        # 3. Generar embeddings
        self.generate_embeddings()
        
        # 4. Poblar ChromaDB
        self.populate_chroma_db()
        
        self.logger.info("Procesamiento completo finalizado")
    
    def search_similar(self, 
                      query_text: str, 
                      n_results: int = 5) -> Dict[str, Any]:
        """
        Busca documentos similares al texto de consulta
        
        Args:
            query_text: Texto de consulta
            n_results: Número de resultados a retornar
            
        Returns:
            Resultados de la búsqueda
        """
        # Generar embedding para la consulta
        query_embedding = self.openaiembeddings_model_client.embed_query(query_text)
        
        # Obtener colección
        collection = self.chroma_client.get_collection(name=self.COLLECTION_NAME)
        
        # Realizar búsqueda
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results



# Ejemplo de uso
if __name__ == "__main__":
    get_chroma = ChromaEmbeddingProcessor()
    get_chroma.run()