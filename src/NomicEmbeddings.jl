module NomicEmbeddings

using CUDA
using cuDNN
using DataDeps
using ONNXRunTime
using JSON
using Unicode
using Unicode: normalize
using DoubleArrayTries
using WordTokenizers
using WordTokenizers: TokenBuffer, isdone, flush!, character, spaces, atoms


include("wordpiece.jl")
include("Nomic.jl")

function __init__()

    # Models
    register(
        DataDep(
            "nomic_embed_text_v1_model",
            "MODEL",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/onnx/model.onnx",
        ),
    )

    register(
        DataDep(
            "nomic_embed_text_v1_model_fp16",
            "MODEL_FP16",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/onnx/model_fp16.onnx",
        ),
    )

    register(
        DataDep(
            "nomic_embed_text_v1_model_quantized",
            "MODEL_QUANTIZED",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/onnx/model_quantized.onnx",
        ),
    )

    # Configs
    # register config 
    register(
        DataDep(
            "nomic_embed_text_v1_config",
            "CONFIG",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/config.json",
        ),
    )



    # register tokenizer
    register(
        DataDep(
            "nomic_embed_text_v1_tokenizer",
            "TOKENIZER",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/tokenizer.json",
        ),
    )

    # register tokenizer config 
    register(
        DataDep(
            "nomic_embed_text_v1_tokenizer_config",
            "TOKENIZER_CONFIG",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/tokenizer_config.json",
        ),
    )

    # register vocab.txt
    register(
        DataDep(
            "nomic_embed_text_v1_vocab",
            "VOBA",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/vocab.txt",
        ),
    )

    # register quantize_config.json
    register(
        DataDep(
            "nomic_embed_text_v1_quantize_config",
            "QUANTIZE_CONFIG",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/quantize_config.json",
        ),
    )

    # register special_tokens_map.json
    register(
        DataDep(
            "nomic_embed_text_v1_special_tokens_map",
            "SPECIAL_TOKEN_MAP",
            "https://huggingface.co/Xenova/nomic-embed-text-v1/resolve/main/special_tokens_map.json",
        ),
    )

end


end # module NomicEmbeddings
