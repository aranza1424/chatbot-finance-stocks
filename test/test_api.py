import sys
from pathlib import Path
import os
from unittest.mock import patch, Mock
# Agregar la carpeta raíz al path de Python
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


with patch('langchain_openai.ChatOpenAI') as mock_chat_openai, \
     patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings, \
     patch('agent.utils.nodes.QueryNodes') as mock_query_nodes:

    # Mock para ChatOpenAI
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Esta es una respuesta de prueba del mock")
    mock_chat_openai.return_value = mock_llm

    # Mock para OpenAIEmbeddings
    mock_embed = Mock()
    mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embed.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_embeddings.return_value = mock_embed

    # Mock para QueryNodes y su colección
    mock_query_nodes_instance = Mock()
    mock_query_nodes.return_value = mock_query_nodes_instance

    # Importar después de aplicar mocks
    from fastapi.testclient import TestClient
    from main import app


# Crear el cliente de pruebas
client = TestClient(app)

class TestAskEndpoint:
    """Tests para el endpoint /ask"""
    
    def test_ask_question_success(self):
        """Test básico: request válido debe retornar respuesta correcta"""
        # Datos de prueba
        test_data = {
            "question": "¿Cómo estás?",
            "thread_id": "test_thread_123"
        }
        
        # Hacer request
        response = client.post("/API/ask", json=test_data)
        
        # Verificaciones
        assert response.status_code == 200
        
        response_data = response.json()
        assert "answer" in response_data
    
    def test_ask_question_empty_question(self):
        """Test con pregunta vacía"""
        test_data = {
            "question": "",
            "thread_id": "test_thread_456"
        }
        
        response = client.post("/API/ask", json=test_data)
        
        # Debe funcionar aunque la pregunta esté vacía
        assert response.status_code == 200

   
    def test_ask_question_missing_fields(self):
        """Test con campos faltantes"""
        # Sin question
        response = client.post("/API/ask", json={"thread_id": "test"})
        assert response.status_code == 422
        
        # Sin thread_id
        response = client.post("/API/ask", json={"question": "test"})
        assert response.status_code == 422
        
        # Vacío
        response = client.post("/API/ask", json={})
        assert response.status_code == 422
    
    def test_ask_question_wrong_data_types(self):
        """Test con tipos de datos incorrectos"""
        # question como número
        test_data = {
            "question": 123,
            "thread_id": "test_thread"
        }
        response = client.post("/API/ask", json=test_data)
        assert response.status_code == 422
        
        # thread_id como número
        test_data = {
            "question": "test question",
            "thread_id": 456
        }
        response = client.post("/API/ask", json=test_data)
        assert response.status_code == 422
    

class TestAppBasics:
    """Tests básicos de la aplicación"""
    
    def test_cors_middleware(self):
        """Verificar que CORS está configurado"""
        # Verificar que el middleware CORS existe
        cors_middleware = None
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware):
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None, "CORS middleware no encontrado"
    
    def test_invalid_endpoint(self):
        """Test de endpoint que no existe"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_wrong_method(self):
        """Test con método HTTP incorrecto"""
        response = client.get("/API/ask")  # GET en lugar de POST
        assert response.status_code == 405  # Method Not Allowed