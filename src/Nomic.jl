
abstract type AbstractEmbedder end

export nomicModel,
    NomicEmbedderModel, reverse_keymap_to_list, extract_added_token, embed, encode

# Most parts of this code are borrowed from transformers.jl and FlashRank.jl

function isinvalid(c)
    if c == '\t' || c == '\n' || c == '\r'
        return false
    end
    c == Char(0) || c == Char(0xfffd) || iscntrl(c)
end

# ignore invalid characters such like U+0000, U+fffd, and Control characters
function invalid(ts)
    isinvalid(ts[]) || return false
    ts.idx += 1
    return true
end

function ischinese(c)
    Char(0x4e00) ≤ c ≤ Char(0x9fff) ||
        Char(0x3400) ≤ c ≤ Char(0x4dbf) ||
        Char(0x20000) ≤ c ≤ Char(0x2a6df) ||
        Char(0x2a700) ≤ c ≤ Char(0x2b73f) ||
        Char(0x2b740) ≤ c ≤ Char(0x2b81f) ||
        Char(0x2b820) ≤ c ≤ Char(0x2ceaf) ||
        Char(0xf900) ≤ c ≤ Char(0xfaff) ||
        Char(0x2f800) ≤ c ≤ Char(0x2fa1f)
end

# separate on chinese characters
function chinese(ts)
    ischinese(ts[]) || return false
    flush!(ts, string(ts[]))
    ts.idx += 1
    return true
end

function isbertpunct(c)
    ispunct(c) ||
        Char(33) ≤ c ≤ Char(47) ||
        Char(58) ≤ c ≤ Char(64) ||
        Char(91) ≤ c ≤ Char(96) ||
        Char(123) ≤ c ≤ Char(126)
end

function bertpunct(ts)
    isbertpunct(ts[]) || return false
    flush!(ts, string(ts[]))
    ts.idx += 1
    return true
end

iscatemn(c) = Base.Unicode.category_code(c) == Base.Unicode.UTF8PROC_CATEGORY_MN
function catemn(ts)
    iscatemn(ts[]) || return false
    ts.idx += 1
    return true
end

#=
bert basic tokenizer pipeline
skip 1. convert to unicode
2. clean text
3. handle chinese character
4. tokenize with white space
5. if lower case : lower, NFD normalize, skip 'Mn' unicode on each tokens
6. split each token with punct and punct remain

=#
function _bert_tokenise(input, ::Val{lower}) where {lower}
    ts = TokenBuffer(lower ? Unicode.normalize(lowercase(input), :NFD) : input)
    while !isdone(ts)
        (lower && catemn(ts)) ||
            invalid(ts) ||
            chinese(ts) ||
            spaces(ts) ||
            bertpunct(ts) ||
            character(ts)
    end
    return ts.tokens
end

"""
    bert_uncased_tokenizer(input)

Google bert tokenizer which do lower case on input before tokenization.
"""
bert_uncased_tokenizer(input) = _bert_tokenise(input, Val(true))

"""
    bert_cased_tokenizer(input)

Google bert tokenizer which remain the case during tokenization. Recommended for multi-lingual data.
"""
bert_cased_tokenizer(input) = _bert_tokenise(input, Val(false))

function extract_added_token(added_token)
    vidx = added_token["id"] + 1
    token = added_token["content"]
    isspecial = added_token["special"]

    added_token["rstrip"] ||
        added_token["lstrip"] && tokenizer_warn(
            "match token `$token` require to match with space on either side but that is not implemented here",
        )
    added_token["single_word"] && tokenizer_warn(
        "match token `$token` does not match inside of a word but that is not implemented here",
    )
    return vidx, token, isspecial
end

function reverse_keymap_to_list(dict)
    vocab_list = Vector{String}(undef, length(dict))
    for (k, v) in dict
        v += 1
        @assert !isassigned(vocab_list, v) "Two word has same index: $(k) and $(vocab_list[v])"
        vocab_list[v] = String(k)
    end
    @assert all(Base.Fix1(isassigned, vocab_list), eachindex(vocab_list)) "There is a gap in the vocabulary"
    return vocab_list
end


extract_and_add_tokens!(::Nothing, _) = nothing
function extract_and_add_tokens!(added_token_list, vocab_list)
    iszero(length(added_token_list)) && return nothing
    added_token_list = sort(added_token_list; by = Base.Fix2(getindex, "id"))
    match_tokens = String[]
    for added_token in added_token_list
        vidx, token, isspecial = extract_added_token(added_token)
        if isspecial
            if vidx > length(vocab_list)
                # special tokens not in the vocab already
                @assert vidx == length(vocab_list) + 1
                push!(vocab_list, token)
            end
            @assert vocab_list[vidx] == token
        else
            n_vocab = length(vocab_list)
            if vidx == n_vocab + 1
                push!(vocab_list, token)
            elseif vidx <= n_vocab
                @assert vocab_list[vidx] == token "Two word has same index: $(token) and $(vocab_list[idx])"
            else
                error("There is a gap in the vocabulary")
            end
        end
        push!(match_tokens, token)
    end
    return match_tokens
end


struct NomicEmbedder <: AbstractEmbedder
    modelPath::String
    device::Symbol
    session::InferenceSession
    specialTokens_map::Dict{String,Any}
    vocabSize::Int
    tokenizer::Any
    tokenizerConfig::Dict{String,Any}
    vocab::Dict{String,Int}
    quantizeConfig::Dict{String,Any}
    config::Dict{String,Any}
    trunc::Int
    wp::WordPiece
    startsym::String
    endsym::String
    padSym::String
end


function nomicModel(device::Symbol, model::String = "nomic")
    modelPath = joinpath(datadep"nomic_embed_text_v1_model", "model.onnx")
    configPath = joinpath(datadep"nomic_embed_text_v1_config", "config.json")
    tokenizerPath = joinpath(datadep"nomic_embed_text_v1_tokenizer", "tokenizer.json")
    tokenizerConfigPath =
        joinpath(datadep"nomic_embed_text_v1_tokenizer_config", "tokenizer_config.json")
    specialTokensMapPath =
        joinpath(datadep"nomic_embed_text_v1_special_tokens_map", "special_tokens_map.json")
    quantizeConfigPath =
        joinpath(datadep"nomic_embed_text_v1_quantize_config", "quantize_config.json")
    vocabPath = joinpath(datadep"nomic_embed_text_v1_vocab", "vocab.txt")
    specialTokensMap = JSON.parsefile(specialTokensMapPath)
    tokenizer = JSON.parsefile(tokenizerPath)
    vocab = readlines(vocabPath)
    vocabSize = length(vocab)
    tokenizerConfig = JSON.parsefile(tokenizerConfigPath)
    quantizeConfig = JSON.parsefile(quantizeConfigPath)
    config = JSON.parsefile(configPath)
    modelSession = ONNXRunTime.load_inference(modelPath; execution_provider = device)

    modelConfig = tokenizer["model"]
    vocabList = reverse_keymap_to_list(modelConfig["vocab"])
    extract_and_add_tokens!(tokenizer["added_tokens"], vocabList)
    vocab = Dict(k => i - 1 for (i, k) in enumerate(vocabList))

    wp = WordPiece(
        vocabList,
        modelConfig["unk_token"];
        max_char = modelConfig["max_input_chars_per_word"],
        subword_prefix = modelConfig["continuing_subword_prefix"],
    )

    @assert get(tokenizer["normalizer"], "lowerer_case", true) "Tokenizer must be lowercased. Model implementation is not compatible."

    trunc = get(tokenizer, "truncation", nothing) |> x -> isnothing(x) ? 512 : x

    enc = NomicEmbedder(
        modelPath,
        device,
        modelSession,
        specialTokensMap,
        vocabSize,
        tokenizer,
        tokenizerConfig,
        vocab,
        quantizeConfig,
        config,
        trunc,
        wp,
        "[CLS]",
        "[SEP]",
        "[PAD]",
    )
    return (enc, modelSession)
end



abstract type AbstractEmbedderModel end

struct NomicEmbedderModel <: AbstractEmbedderModel
    alias::Symbol
    encoder::NomicEmbedder
    session::InferenceSession
end

"""
    EmbedResult{T <: Real}

The result of embedding passages.

# Fields
- `embeddings::AbstractArray{T}`: The embeddings of the passages. With property `embeddings` as column-major matrix of size `(batch_size, embedding_dimension)`.
- `elapsed::Float64`: The time taken to embed the passages.
"""
struct EmbedResult{T<:Real}
    embeddings::AbstractArray{T}
    elapsed::Float64
end
function Base.show(io::IO, result::EmbedResult)
    dump(io, result; maxdepth = 1)
end


"""
    tokenize(enc::NomicEmbedderModel, text::AbstractString;
        add_special_tokens::Bool = true, add_end_token::Bool = true, token_ids::Bool = false,
        max_tokens::Union{Nothing, Int} = enc.trunc)

Tokenizes the text and returns the tokens or token IDs (to skip looking up the IDs twice).

# Arguments
- `add_special_tokens::Bool = true`: Add special tokens at the beginning and end of the text.
- `add_end_token::Bool = true`: Add end token at the end of the text.
- `token_ids::Bool = false`: If true, return the token IDs directly. Otherwise, return the tokens.
- `max_tokens::Union{Nothing, Int} = enc.trunc`: The maximum number of tokens to return (usually defined by the model).
"""
function tokenize(
    enc::NomicEmbedder,
    text::AbstractString;
    add_special_tokens::Bool = true,
    add_end_token::Bool = true,
    token_ids::Bool = false,
    max_tokens::Union{Nothing,Int} = enc.trunc,
)
    tokens = token_ids ? Int[] : String[]
    if add_special_tokens
        token = token_ids ? enc.vocab[enc.startsym] : enc.startsym
        push!(tokens, token)
    end
    for token in bert_uncased_tokenizer(text)
        append!(tokens, enc.wp(token; token_ids))
    end
    if !isnothing(max_tokens) && length(tokens) > (max_tokens - 1)
        tokens = tokens[1:(max_tokens-1)]
    end
    if add_special_tokens || add_end_token
        token = token_ids ? enc.vocab[enc.endsym] : enc.endsym
        push!(tokens, token)
    end
    return tokens
end

"""
    encode(enc::NomicEmbedderModel, text::String; add_special_tokens::Bool = true,
        max_tokens::Int = enc.trunc, split_instead_trunc::Bool = false)

Encodes the text and returns the token IDs, token type IDs, and attention mask.

We enforce `max_tokens` to be a concrete number here to be able to do `split_instead_trunc`.
`split_instead_trunc` splits any long sequences into several smaller ones.
"""
function encode(
    enc::NomicEmbedder,
    text::String;
    add_special_tokens::Bool = true,
    max_tokens::Int = enc.trunc,
    split_instead_trunc::Bool = false,
)
    if !split_instead_trunc
        ## Standard run - if text is longer, we truncate it and ignore
        token_ids = tokenize(enc, text; add_special_tokens, token_ids = true, max_tokens)
        # Zero indexed as models are trained for Python
        token_type_ids = zeros(Int, length(token_ids))
        attention_mask = ones(Int, length(token_ids))
    else
        ## Split run - if text is longer, we split it into multiple chunks and encode them separately
        ## Only possible with a single string to know where the chunks belong to
        ## tokenize without special tokens at first
        token_ids = tokenize(
            enc,
            text;
            add_special_tokens = false,
            token_ids = true,
            max_tokens = nothing,
        )
        ## determine correct chunk size
        start_token = enc.vocab[enc.startsym]
        end_token = enc.vocab[enc.endsym]
        chunk_size = max_tokens - 2 * add_special_tokens
        itr = Iterators.partition(token_ids, chunk_size)
        num_chunks = length(itr)
        ## split vector in several
        mat_token_ids = zeros(Int, max_tokens, num_chunks)
        token_type_ids = zeros(Int, max_tokens, num_chunks)
        attention_mask = zeros(Int, max_tokens, num_chunks)
        @inbounds for (i, chunk) in enumerate(itr)
            if add_special_tokens
                mat_token_ids[1, i] = start_token
                attention_mask[1, i] = 1
            end
            for ri in eachindex(chunk)
                ## if special token, we shift all items by 1 down
                row_idx = add_special_tokens ? ri + 1 : ri
                mat_token_ids[row_idx, i] = chunk[ri]
                attention_mask[row_idx, i] = 1
            end
            if add_special_tokens
                row_idx = 2 + length(chunk)
                mat_token_ids[row_idx, i] = end_token
                attention_mask[row_idx, i] = 1
            end
        end
        token_ids = mat_token_ids
    end
    return token_ids, token_type_ids, attention_mask
end

function encode(
    enc::NomicEmbedder,
    query::AbstractString,
    passage::AbstractString;
    add_special_tokens::Bool = true,
)
    ## Tokenize texts
    token_ids1 = tokenize(enc, query; add_special_tokens, token_ids = true)
    token_ids2 = tokenize(
        enc,
        passage;
        add_special_tokens = false,
        add_end_token = add_special_tokens,
        token_ids = true,
    )
    token_type_ids = vcat(zeros(Int, length(token_ids1)), ones(Int, length(token_ids2)))
    token_ids = vcat(token_ids1, token_ids2)

    ## check if we exceed truncation
    if !isnothing(enc.trunc) && (length(token_ids)) > enc.trunc
        token_ids = first(token_ids, enc.trunc)
        ## add [SEP] token
        token_ids[end] = enc.vocab[enc.endsym]
        token_type_ids = first(token_type_ids, enc.trunc)
    end

    # Zero indexed as models are trained for Python
    attention_mask = ones(Int, length(token_ids))
    return token_ids, token_type_ids, attention_mask
end

function encode(
    enc::NomicEmbedder,
    query::AbstractString,
    passages::AbstractVector{<:AbstractString};
    add_special_tokens::Bool = true,
)

    ## tokenize query, it will be repeated
    token_ids1 = tokenize(enc, query; add_special_tokens, token_ids = true)

    tokens_ids2_vec = [
        tokenize(
            enc,
            passage;
            add_special_tokens = false,
            add_end_token = add_special_tokens,
            token_ids = true,
        ) for passage in passages
    ]
    len_ =
        maximum(length, tokens_ids2_vec) + length(token_ids1) |>
        x -> isnothing(enc.trunc) ? x : min(x, enc.trunc)

    ## Assumes that padding is done with token ID 0
    token_ids = zeros(Int, len_, length(passages))
    token_type_ids = zeros(Int, len_, length(passages))
    attention_mask = zeros(Int, len_, length(passages))

    ## Encode to token IDS
    token_ids1_len = length(token_ids1)
    @inbounds for j in eachindex(tokens_ids2_vec)
        token_ids[1:token_ids1_len, j] .= token_ids1
        attention_mask[1:token_ids1_len, j] .= 1

        tokens_ids2 = tokens_ids2_vec[j]
        for i in eachindex(tokens_ids2)
            if token_ids1_len + i > len_
                break
            elseif token_ids1_len + i == len_
                ## give [SEP] token
                token_ids[token_ids1_len+i, j] = enc.vocab[enc.endsym]
            else
                ## fill the tokens
                token_ids[token_ids1_len+i, j] = tokens_ids2[i]
            end
            token_type_ids[token_ids1_len+i, j] = 1
            attention_mask[token_ids1_len+i, j] = 1
        end
    end
    return token_ids, token_type_ids, attention_mask
end

"""
    embed(
        embedder::NomicEmbedderModel, passages::AbstractVector{<:AbstractString})

Embeds `passages` using the given `embedder` model.

# Arguments:
- `embedder::NomicEmbedderModel`: The embedder model to use.
- `passages::AbstractVector{<:AbstractString}`: The passages to embed.

# Returns
- `EmbedResult`: The embeddings of the passages. With property `embeddings` as column-major matrix of size `(batch_size, embedding_dimension)`.

# Example
```julia
model = NomicEmbedderModel(:tiny_embed)
result = embed(model, ["Hello, how are you?", "How is it going?"])
result.embeddings # 312x2 matrix of Float32
```
"""

function embed(embedder::NomicEmbedderModel, passages::AbstractVector{<:AbstractString})
    t = @elapsed begin
        token_ids, token_type_ids, attention_mask = encode(embedder.encoder, passages)
        ## transpose as the model expects row-major
        ## TODO: investigate pre-warming the session with padded inputs
        ## TODO: investigate performnance on materialized inputs
        onnx_input = Dict(
            "input_ids" => token_ids',
            "token_type_ids" => token_type_ids',
            "attention_mask" => attention_mask',
        )
        out = embedder.session(onnx_input)
        ## Permute dimensions to return column-major embeddings, ie, batch-size X embedding-size
        embeddings = out["last_hidden_state"]
    end
    EmbedResult(embeddings, t)
end

"""
    embed(
        embedder::NomicEmbedderModel, passage::AbstractString; split_instead_trunc::Bool = false)

Embeds a single `passage`. 

If passage is too long for the model AND `split_instead_trunc` is true, the passage is split into several smaller chunks of size `embedder.encoder.trunc` and embedded separately.
"""
function embed(
    embedder::NomicEmbedderModel,
    passage::AbstractString;
    split_instead_trunc::Bool = false,
)
    t = @elapsed begin
        token_ids, token_type_ids, attention_mask =
            encode(embedder.encoder, passage; split_instead_trunc)
        ## transpose as the model expects row-major
        onnx_input = Dict(
            "input_ids" => token_ids',
            "token_type_ids" => token_type_ids',
            "attention_mask" => attention_mask',
        )
        out = embedder.session(onnx_input)
        ## Permute dimensions to return column-major embeddings, ie, batch-size X embedding-size
        embeddings = out["last_hidden_state"]
    end
    EmbedResult(embeddings, t)
end

function (embedder::NomicEmbedderModel)(
    passages::Union{AbstractString,AbstractVector{<:AbstractString}};
    kwargs...,
)
    embed(embedder, passages; kwargs...)
end

function encode(
    enc::NomicEmbedder,
    passages::AbstractVector{<:AbstractString};
    add_special_tokens::Bool = true,
)
    tokens_vec = [
        tokenize(enc, passage; add_special_tokens = true, token_ids = true) for
        passage in passages
    ]
    max_len =
        maximum(length, tokens_vec) |> x -> isnothing(enc.trunc) ? x : min(x, enc.trunc)

    ## Assumes that padding is done with token ID 0
    token_ids = zeros(Int, max_len, length(passages))
    # Zero indexed as models are trained for Python
    token_type_ids = zeros(Int, max_len, length(passages))
    attention_mask = zeros(Int, max_len, length(passages))

    ## Encode to token IDS
    @inbounds for j in eachindex(tokens_vec)
        tokens = tokens_vec[j]
        for i in eachindex(tokens)
            if i > max_len
                break
            elseif i == max_len
                ## give [SEP] token
                token_ids[i, j] = enc.vocab[enc.endsym]
            else
                ## fill the tokens
                token_ids[i, j] = tokens[i]
            end
            attention_mask[i, j] = 1
        end
    end
    return token_ids, token_type_ids, attention_mask
end

