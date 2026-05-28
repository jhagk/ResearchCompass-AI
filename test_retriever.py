from rag.retriever import get_qa_chain

qa_chain = get_qa_chain()

result = qa_chain.invoke({
    "query": "What are the open research problems in image classification?"
})

print("Answer:")
print(result["result"])
print("\n---")
print("Source papers used:")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata['title']}")