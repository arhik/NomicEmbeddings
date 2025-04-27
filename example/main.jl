using NomicEmbeddings
using LinearAlgebra
using Statistics

(encoder, session) = nomicModel(:cuda)
embedder = NomicEmbedderModel(:nomic, encoder, session)

function meanPool(encoder, session, text)
    token_ids, token_type_ids, attention_mask = encode(encoder, text)
    onnx_input = Dict(
        "input_ids" => token_ids',
        "token_type_ids" => token_type_ids',
        "attention_mask" => attention_mask',
    )
    out = embedder.session(onnx_input)
    embeddings = permutedims(out["last_hidden_state"], (3, 2, 1))
    expandedAttenMask = permutedims(repeat(attention_mask, inner=(1, 1, size(embeddings, 1))), (3, 1, 2))
    embeddings .*= expandedAttenMask
    embs = sum(embeddings, dims=2)./sum(expandedAttenMask, dims=2)
    return embs./sqrt(sum(embs.^2, dims=1))
end

embs = meanPool(encoder, session, "Hello!")

sentences = ["search_query: What is TSNE?", "search_query: Who is Laurens van der Maaten?"]
embs = meanPool(encoder, session, sentences)
