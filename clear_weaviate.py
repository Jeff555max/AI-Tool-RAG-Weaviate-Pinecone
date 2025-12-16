"""
Очистка Weaviate
"""
import weaviate
from config.settings import settings

client = weaviate.connect_to_wcs(
    cluster_url=settings.WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY)
)

class_name = "RAGDocument"
if client.collections.exists(class_name):
    print(f"Удаление коллекции '{class_name}'...")
    client.collections.delete(class_name)
    print("Коллекция удалена!")
else:
    print("Коллекция не существует")

client.close()
