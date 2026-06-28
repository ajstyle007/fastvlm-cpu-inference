#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_NEW_TOKENS 512

float* load_embeddings(const char* path, int* n_tokens, int* n_embd) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
    fread(n_tokens, sizeof(int32_t), 1, f);
    fread(n_embd,   sizeof(int32_t), 1, f);
    float* data = malloc((*n_tokens) * (*n_embd) * sizeof(float));
    fread(data, sizeof(float), (*n_tokens) * (*n_embd), f);
    fclose(f);
    fprintf(stderr, "Embeddings: %d x %d\n", *n_tokens, *n_embd);
    return data;
}

llama_token* tokenize_str(const struct llama_vocab* vocab,
                          const char* text, int* n_out) {
    int n = -llama_tokenize(vocab, text, strlen(text), NULL, 0, false, true);
    llama_token* toks = malloc(n * sizeof(llama_token));
    llama_tokenize(vocab, text, strlen(text), toks, n, false, true);
    *n_out = n;
    return toks;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <embeddings.bin> <prompt>\n", argv[0]);
        return 1;
    }

    const char* model_path  = argv[1];
    const char* embd_path   = argv[2];
    const char* user_prompt = argv[3];

    // Load image embeddings
    int n_img, n_embd;
    float* img_embd = load_embeddings(embd_path, &n_img, &n_embd);

    // Init model
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    struct llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 2048;
    cp.n_batch  = 512;
    cp.n_ubatch = 512;
    struct llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const struct llama_vocab* vocab = llama_model_get_vocab(model);

    // Build prompt parts
    // Structure: [prefix] [image embeddings] [suffix]
    // prefix = system + start of user turn
    // suffix = newline + user text + end of turn + assistant start
    char prefix[1024];
    snprintf(prefix, sizeof(prefix),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
    );

    char suffix[2048];
    snprintf(suffix, sizeof(suffix),
        "\n%s<|im_end|>\n<|im_start|>assistant\n",
        user_prompt
    );

    int n_prefix, n_suffix;
    llama_token* prefix_toks = tokenize_str(vocab, prefix, &n_prefix);
    llama_token* suffix_toks = tokenize_str(vocab, suffix, &n_suffix);

    fprintf(stderr, "Prefix: %d tokens | Image: %d tokens | Suffix: %d tokens\n",
            n_prefix, n_img, n_suffix);

    int cur_pos = 0;

    // Batch 1: prefix tokens
    {
        struct llama_batch b = llama_batch_init(n_prefix, 0, 1);
        b.n_tokens = n_prefix;
        for (int i = 0; i < n_prefix; i++) {
            b.token[i]     = prefix_toks[i];
            b.pos[i]       = cur_pos++;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 0;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "Failed prefix decode\n"); return 1;
        }
        llama_batch_free(b);
    }

    // Batch 2: image embeddings
    {
        struct llama_batch b = llama_batch_init(n_img, n_embd, 1);
        b.n_tokens = n_img;
        for (int i = 0; i < n_img; i++) {
            memcpy(b.embd + i * n_embd,
                   img_embd + i * n_embd,
                   n_embd * sizeof(float));
            b.pos[i]       = cur_pos++;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 0;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "Failed image decode\n"); return 1;
        }
        llama_batch_free(b);
    }

    // Batch 3: suffix tokens
    {
        struct llama_batch b = llama_batch_init(n_suffix, 0, 1);
        b.n_tokens = n_suffix;
        for (int i = 0; i < n_suffix; i++) {
            b.token[i]     = suffix_toks[i];
            b.pos[i]       = cur_pos++;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = (i == n_suffix - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "Failed suffix decode\n"); return 1;
        }
        llama_batch_free(b);
    }

    fprintf(stderr, "Context: %d tokens total. Generating...\n\n", cur_pos);

    // Get stop tokens
    llama_token eos = llama_vocab_eos(vocab);
    int n_tmp;
    llama_token* tmp  = tokenize_str(vocab, "<|im_end|>", &n_tmp);
    llama_token im_end = tmp[0];
    free(tmp);

    // Sampler
    struct llama_sampler* sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // Generation loop
    for (int i = 0; i < MAX_NEW_TOKENS; i++) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);

        if (tok == eos || tok == im_end) break;

        char piece[128] = {0};
        llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
        printf("%s", piece);
        fflush(stdout);

        llama_sampler_accept(sampler, tok);

        struct llama_batch next = llama_batch_init(1, 0, 1);
        next.n_tokens      = 1;
        next.token[0]      = tok;
        next.pos[0]        = cur_pos++;
        next.n_seq_id[0]   = 1;
        next.seq_id[0][0]  = 0;
        next.logits[0]     = 1;
        if (llama_decode(ctx, next) != 0) break;
        llama_batch_free(next);
    }
    printf("\n");

    // Cleanup
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(img_embd);
    free(prefix_toks);
    free(suffix_toks);
    return 0;
}