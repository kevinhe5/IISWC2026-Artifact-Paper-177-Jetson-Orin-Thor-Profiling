#!/usr/bin/env python3
"""Read GGUF model metadata and predict memory usage."""
import json
import struct
import sys
import os
from pathlib import Path

QUANT_NAMES = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
    8: 'Q8_0', 15: 'Q4_K_M', 16: 'Q5_K_M', 17: 'Q6_K', 18: 'Q8_0'
}

# Bits per element for different quantization types
# For weights: includes quantization overhead (scales, etc.)
# For KV cache: llama.cpp uses these for cache_type_k/cache_type_v
QUANT_BITS = {
    'F32': 32,
    'F16': 16,
    'BF16': 16,
    'Q8_0': 8.5,    # 8 bits + 0.5 for block scales
    'Q6_K': 6.5625, # 6 bits + overhead
    'Q5_K_M': 5.5,
    'Q5_K_S': 5.5,
    'Q5_1': 6,
    'Q5_0': 5.5,
    'Q4_K_M': 4.875,
    'Q4_K_S': 4.5,
    'Q4_1': 5,
    'Q4_0': 4.5,
    'Q3_K_M': 3.875,
    'Q2_K': 2.625,
    'IQ4_NL': 4.5,
    'IQ4_XS': 4.25,
    'type_-1': 4.5  # fallback
}

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def read_value(f, vtype):
    readers = {
        0: lambda: struct.unpack('<B', f.read(1))[0],
        1: lambda: struct.unpack('<b', f.read(1))[0],
        2: lambda: struct.unpack('<H', f.read(2))[0],
        3: lambda: struct.unpack('<h', f.read(2))[0],
        4: lambda: struct.unpack('<I', f.read(4))[0],
        5: lambda: struct.unpack('<i', f.read(4))[0],
        6: lambda: struct.unpack('<f', f.read(4))[0],
        7: lambda: struct.unpack('<B', f.read(1))[0] != 0,
        8: lambda: read_string(f),
        10: lambda: struct.unpack('<Q', f.read(8))[0],
        11: lambda: struct.unpack('<q', f.read(8))[0],
        12: lambda: struct.unpack('<d', f.read(8))[0],
    }
    if vtype == 9:  # array
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    return readers.get(vtype, lambda: None)()

def read_metadata(filepath: str) -> dict:
    """Read GGUF file and return metadata."""
    with open(filepath, 'rb') as f:
        if f.read(4) != b'GGUF':
            return {"error": "Not a valid GGUF file"}

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]

        metadata = {}
        for _ in range(kv_count):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            metadata[key] = read_value(f, vtype)

    arch = metadata.get('general.architecture', 'unknown')
    quant = metadata.get('general.file_type', -1)
    vocab = metadata.get(f'{arch}.vocab_size', metadata.get('tokenizer.ggml.tokens', []))

    return {
        "name": metadata.get('general.name', metadata.get('general.basename', 'Unknown')),
        "architecture": arch,
        "layers": metadata.get(f'{arch}.block_count'),
        "context_length": metadata.get(f'{arch}.context_length'),
        "embedding_dim": metadata.get(f'{arch}.embedding_length'),
        "attention_heads": metadata.get(f'{arch}.attention.head_count'),
        "kv_heads": metadata.get(f'{arch}.attention.head_count_kv'),
        "ffn_dim": metadata.get(f'{arch}.feed_forward_length'),
        "vocab_size": len(vocab) if isinstance(vocab, list) else vocab,
        "quantization": QUANT_NAMES.get(quant, f'type_{quant}'),
        "tensor_count": tensor_count,
        "version": version
    }


def _ensure_int(val, default=1):
    """Ensure value is an integer, handling lists and None."""
    if val is None:
        return default
    if isinstance(val, list):
        return val[0] if val else default
    return int(val)


def calculate_memory(info: dict, filepath: str, prompt_tokens: int = 100,
                     gen_tokens: int = 100, chunk_size: int = 32,
                     context_size: int = 256, kv_cache_dtype: str = 'F16',
                     activation_dtype: str = 'F16') -> dict:
    """Calculate predicted memory usage for different components."""

    # Get model params (ensure all are integers, some GGUF files return lists)
    layers = _ensure_int(info.get('layers'), 1)
    embed_dim = _ensure_int(info.get('embedding_dim'), 512)
    n_heads = _ensure_int(info.get('attention_heads'), 8)
    n_kv_heads = _ensure_int(info.get('kv_heads'), n_heads)
    ffn_dim = _ensure_int(info.get('ffn_dim'), embed_dim * 4)
    vocab_size = _ensure_int(info.get('vocab_size'), 32000)
    quant = info.get('quantization', 'Q4_K_M')
    head_dim = embed_dim // n_heads

    # 1. Model weights - use actual file size
    model_weights_mb = os.path.getsize(filepath) / (1024 * 1024)

    # Theoretical weight size (accounts for GQA: K,V use fewer heads)
    q_params = embed_dim * (n_heads * head_dim)      # Q projection
    k_params = embed_dim * (n_kv_heads * head_dim)   # K projection (GQA)
    v_params = embed_dim * (n_kv_heads * head_dim)   # V projection (GQA)
    o_params = (n_heads * head_dim) * embed_dim      # O projection
    attn_params = q_params + k_params + v_params + o_params
    ffn_params = 3 * embed_dim * ffn_dim  # gate, up, down
    params_per_layer = attn_params + ffn_params
    total_params = layers * params_per_layer + vocab_size * embed_dim * 2

    bits_per_weight = QUANT_BITS.get(quant, 4.5)
    theoretical_weights_mb = (total_params * bits_per_weight / 8) / (1024 * 1024)

    # 2. KV Cache: 2 (K+V) * layers * kv_heads * head_dim * context_length
    # KV cache dtype can be quantized too (F16, Q8_0, Q4_0, etc.)
    kv_bits = QUANT_BITS.get(kv_cache_dtype, 16)  # default F16
    kv_bytes_per_elem = kv_bits / 8
    kv_cache_bytes = 2 * layers * n_kv_heads * head_dim * context_size * kv_bytes_per_elem
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)

    # 3. Intermediate activations (configurable, llama.cpp typically uses F16 on CUDA)
    act_bits = QUANT_BITS.get(activation_dtype, 16)  # default F16
    act_bytes = act_bits / 8

    # Prefill: process chunk_size tokens at once
    prefill_seq = min(prompt_tokens, chunk_size)
    prefill_q = prefill_seq * n_heads * head_dim * act_bytes
    prefill_k = prefill_seq * n_kv_heads * head_dim * act_bytes
    prefill_v = prefill_seq * n_kv_heads * head_dim * act_bytes
    prefill_attn_scores = n_heads * prefill_seq * prefill_seq * act_bytes
    prefill_ffn = prefill_seq * ffn_dim * act_bytes
    prefill_activations_mb = (prefill_q + prefill_k + prefill_v + prefill_attn_scores + prefill_ffn) / (1024 * 1024)

    # Decode: 1 token at a time
    decode_q = 1 * n_heads * head_dim * act_bytes
    decode_k = 1 * n_kv_heads * head_dim * act_bytes
    decode_v = 1 * n_kv_heads * head_dim * act_bytes
    decode_attn_scores = n_heads * 1 * context_size * act_bytes
    decode_ffn = 1 * ffn_dim * act_bytes
    decode_activations_mb = (decode_q + decode_k + decode_v + decode_attn_scores + decode_ffn) / (1024 * 1024)

    # 4. Output logits: vocab_size * 4 bytes (F32 for softmax precision)
    output_logits_mb = (vocab_size * 4) / (1024 * 1024)

    # 5. Scratch/compute buffers
    scratch_mb = max(32, min(256, model_weights_mb * 0.1))

    # Total estimates
    total_prefill_mb = model_weights_mb + kv_cache_mb + prefill_activations_mb + output_logits_mb + scratch_mb
    total_decode_mb = model_weights_mb + kv_cache_mb + decode_activations_mb + output_logits_mb + scratch_mb

    return {
        "model_weights_mb": round(model_weights_mb, 2),
        "theoretical_weights_mb": round(theoretical_weights_mb, 2),
        "kv_cache_mb": round(kv_cache_mb, 2),
        "kv_cache_details": {
            "formula": f"2 * {layers}L * {n_kv_heads}kv * {head_dim}d * {context_size}ctx * {kv_bytes_per_elem:.1f}B ({kv_cache_dtype})",
            "bytes": int(kv_cache_bytes)
        },
        "prefill_activations_mb": round(prefill_activations_mb, 2),
        "prefill_details": {
            "seq_len": prefill_seq,
            "q_mb": round(prefill_q / (1024*1024), 4),
            "k_mb": round(prefill_k / (1024*1024), 4),
            "v_mb": round(prefill_v / (1024*1024), 4),
            "attn_scores_mb": round(prefill_attn_scores / (1024*1024), 4),
            "ffn_mb": round(prefill_ffn / (1024*1024), 4)
        },
        "decode_activations_mb": round(decode_activations_mb, 2),
        "decode_details": {
            "q_mb": round(decode_q / (1024*1024), 4),
            "k_mb": round(decode_k / (1024*1024), 4),
            "v_mb": round(decode_v / (1024*1024), 4),
            "attn_scores_mb": round(decode_attn_scores / (1024*1024), 4),
            "ffn_mb": round(decode_ffn / (1024*1024), 4)
        },
        "output_logits_mb": round(output_logits_mb, 2),
        "scratch_buffer_mb": round(scratch_mb, 2),
        "total_prefill_mb": round(total_prefill_mb, 2),
        "total_decode_mb": round(total_decode_mb, 2),
        "params": {
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "chunk_size": chunk_size,
            "context_size": context_size,
            "kv_cache_dtype": kv_cache_dtype,
            "kv_bits": kv_bits,
            "activation_dtype": activation_dtype,
            "act_bits": act_bits
        }
    }

def print_info(info: dict, mem: dict = None):
    """Print model info in a formatted box."""
    name = info.get('name', 'Unknown')[:34]
    print(f"  +-------------------------------------------+")
    print(f"  | MODEL: {name:<34} |")
    print(f"  +-------------------------------------------+")
    print(f"  |  Arch:        {str(info.get('architecture', 'N/A')):<27} |")
    print(f"  |  Layers:      {str(info.get('layers', 'N/A')):<27} |")
    print(f"  |  Context:     {str(info.get('context_length', 'N/A')):<27} |")
    print(f"  |  Embedding:   {str(info.get('embedding_dim', 'N/A')):<27} |")
    print(f"  |  Heads:       {str(info.get('attention_heads', 'N/A')):<27} |")
    print(f"  |  KV Heads:    {str(info.get('kv_heads', 'N/A')):<27} |")
    print(f"  |  FFN:         {str(info.get('ffn_dim', 'N/A')):<27} |")
    print(f"  |  Vocab:       {str(info.get('vocab_size', 'N/A')):<27} |")
    print(f"  |  Quant:       {str(info.get('quantization', 'N/A')):<27} |")
    print(f"  |  Tensors:     {str(info.get('tensor_count', 'N/A')):<27} |")
    print(f"  +-------------------------------------------+")

    if mem:
        p = mem['params']
        pf = mem['prefill_details']
        dc = mem['decode_details']
        print()
        print(f"  +-------------------------------------------+")
        print(f"  | MEMORY PREDICTION                         |")
        print(f"  +-------------------------------------------+")
        print(f"  | prompt={p['prompt_tokens']}, gen={p['gen_tokens']}, ctx={p['context_size']}, chunk={p['chunk_size']}")
        print(f"  | kv_cache={p['kv_cache_dtype']} ({p['kv_bits']} bits), activations={p['activation_dtype']} ({p['act_bits']} bits)")
        print(f"  +-------------------------------------------+")
        print(f"  |  Model Weights:      {mem['model_weights_mb']:>8.2f} MB         |")
        print(f"  |  KV Cache:           {mem['kv_cache_mb']:>8.2f} MB         |")
        print(f"  |    {mem['kv_cache_details']['formula']}")
        print(f"  +-------------------------------------------+")
        print(f"  |  PREFILL Activations: {mem['prefill_activations_mb']:>7.2f} MB         |")
        print(f"  |    Q: {pf['q_mb']:.4f}  K: {pf['k_mb']:.4f}  V: {pf['v_mb']:.4f} MB")
        print(f"  |    Attn scores: {pf['attn_scores_mb']:.4f}  FFN: {pf['ffn_mb']:.4f} MB")
        print(f"  +-------------------------------------------+")
        print(f"  |  DECODE Activations:  {mem['decode_activations_mb']:>7.2f} MB         |")
        print(f"  |    Q: {dc['q_mb']:.4f}  K: {dc['k_mb']:.4f}  V: {dc['v_mb']:.4f} MB")
        print(f"  |    Attn scores: {dc['attn_scores_mb']:.4f}  FFN: {dc['ffn_mb']:.4f} MB")
        print(f"  +-------------------------------------------+")
        print(f"  |  Output Logits:      {mem['output_logits_mb']:>8.2f} MB         |")
        print(f"  |  Scratch Buffers:    {mem['scratch_buffer_mb']:>8.2f} MB (est)   |")
        print(f"  +-------------------------------------------+")
        print(f"  |  TOTAL PREFILL:      {mem['total_prefill_mb']:>8.2f} MB         |")
        print(f"  |  TOTAL DECODE:       {mem['total_decode_mb']:>8.2f} MB         |")
        print(f"  +-------------------------------------------+")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Read GGUF model metadata and predict memory usage')
    parser.add_argument('model_path', help='Path to GGUF model file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--memory', '-m', action='store_true', help='Show memory prediction')
    parser.add_argument('--prompt-tokens', type=int, default=100, help='Number of prompt tokens')
    parser.add_argument('--gen-tokens', type=int, default=100, help='Number of generation tokens')
    parser.add_argument('--chunk-size', type=int, default=32, help='Chunk size (tokens per forward pass)')
    parser.add_argument('--context-size', type=int, default=256, help='Context size')
    parser.add_argument('--kv-cache-dtype', type=str, default='F16', help='KV cache dtype (F16, F32, Q8_0, Q4_0, etc.)')
    parser.add_argument('--activation-dtype', type=str, default='F16', help='Activation dtype (F16, F32)')
    parser.add_argument('--config', '-c', type=str, help='Load settings from config.json')
    parser.add_argument('--ctx', type=int, help='Override context size (use instead of config value)')
    parser.add_argument('--chunk', type=int, help='Override chunk size (use instead of config value)')

    args = parser.parse_args()

    # Load from config file if specified
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
            bench = cfg.get('benchmark', {})
            args.prompt_tokens = bench.get('prompt_tokens', args.prompt_tokens)
            args.gen_tokens = bench.get('gen_tokens', args.gen_tokens)
            args.chunk_size = bench.get('chunk_size', args.chunk_size)
            # Only use config context_size if --ctx not specified
            if args.ctx is None:
                ctx_from_config = bench.get('context_size', args.context_size)
                # Handle "auto" or other non-integer values
                if isinstance(ctx_from_config, int):
                    args.context_size = ctx_from_config
                # else keep default
            args.kv_cache_dtype = bench.get('kv_cache_dtype', args.kv_cache_dtype)
            args.activation_dtype = bench.get('activation_dtype', args.activation_dtype)

    # --ctx and --chunk override config values
    if args.ctx is not None:
        args.context_size = args.ctx
    if args.chunk is not None:
        args.chunk_size = args.chunk

    if args.config:
        print(f"  [Config loaded: prompt={args.prompt_tokens}, gen={args.gen_tokens}, "
              f"ctx={args.context_size}, chunk={args.chunk_size}, kv={args.kv_cache_dtype}, act={args.activation_dtype}]")

    info = read_metadata(args.model_path)

    mem = None
    if args.memory or args.config:
        mem = calculate_memory(
            info, args.model_path,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens,
            chunk_size=args.chunk_size,
            context_size=args.context_size,
            kv_cache_dtype=args.kv_cache_dtype,
            activation_dtype=args.activation_dtype
        )

    if args.json:
        output = {"model_info": info}
        if mem:
            output["memory_prediction"] = mem
        print(json.dumps(output, indent=2))
    else:
        print_info(info, mem)

if __name__ == "__main__":
    main()
