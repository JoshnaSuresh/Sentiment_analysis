from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://localhost:9200",
    verify_certs=False  # dev only
)

try:
    print("📡 Connecting...")
    info = es.info()
    es.ping()
    print("✅ Connected to ES:", info['version'])
except Exception as e:
    print("💥 Error:", e)
