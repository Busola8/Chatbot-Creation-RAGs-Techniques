from app.ingest import chunk_text
def test_chunk():
    s = ' '.join(str(i) for i in range(1000))
    chunks = chunk_text(s, chunk_size=100, overlap=20)
    assert len(chunks) > 0
