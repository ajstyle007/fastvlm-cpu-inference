// fastvlm_infer.c
// Compile: gcc fastvlm_infer.c -o fastvlm_infer \
//   -I./llama.cpp/include \
//   -L./llama.cpp/build/bin \
//   -lllama -lggml -lggml-base -lm -lstdc++ \
//   -Wl,-rpath,./llama.cpp/build/bin

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define N_IMG_TOKENS  256
#define N_EMBD        896
#define MAX_NEW_TOKENS 512

// Read embeddings binary: [int32 n_tokens][int32 n_embd][float32 * n_tokens * n_embd]
float* load_embeddings(const char* path, int* n_tokens, int* n_embd) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open embeddings: %s\n", path); exit(1); }
    fread(n_tokens, sizeof(int32_t), 1, f);
    fread(n_embd,   sizeof(int32_t), 1, f);
    float* data = malloc((*n_tokens) * (*n_embd) * sizeof(float));
    fread(data, sizeof(float), (*n_tokens) * (*n_embd), f);
    fclose(f);
    fprintf(stderr, "Loaded embeddings: %d tokens x %d dims\n", *n_tokens, *n_embd);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <embeddings.bin> <prompt_text>\n", argv[0]);
        return 1;
    }

    const char* model_path  = argv[1];
    const char* embd_path   = argv[2];
    const char* prompt_text = argv[3];

    // ── Load embeddings ───────────────────────────────────────────────────────
    int n_img_tokens, n_embd;
    float* img_embd = load_embeddings(embd_path, &n_img_tokens, &n_embd);

    // ── Init llama.cpp ────────────────────────────────────────────────────────
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // offload all to GPU
    struct llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 2048;
    cparams.n_batch  = 512;
    cparams.n_ubatch = 512;
    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const struct llama_vocab* vocab = llama_model_get_vocab(model);

    // ── Build text prompt ─────────────────────────────────────────────────────
    char full_prompt[4096];
    snprintf(full_prompt, sizeof(full_prompt),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n",
        prompt_text
    );

    // Tokenize
    int n_prompt_tokens = -llama_tokenize(vocab, full_prompt, strlen(full_prompt),
                                          NULL, 0, false, true);
    llama_token* prompt_tokens = malloc(n_prompt_tokens * sizeof(llama_token));
    llama_tokenize(vocab, full_prompt, strlen(full_prompt),
                   prompt_tokens, n_prompt_tokens, false, true);

    fprintf(stderr, "Prompt tokens: %d, Image tokens: %d\n",
            n_prompt_tokens, n_img_tokens);

    // ── Batch 1: image embeddings (inject as raw embeddings) ─────────────────
    struct llama_batch img_batch = llama_batch_init(n_img_tokens, n_embd, 1);
    img_batch.n_tokens = n_img_tokens;
    for (int i = 0; i < n_img_tokens; i++) {
        // Copy embedding vector for token i
        memcpy(img_batch.embd + i * n_embd,
               img_embd + i * n_embd,
               n_embd * sizeof(float));
        img_batch.pos[i]        = i;
        img_batch.n_seq_id[i]   = 1;
        img_batch.seq_id[i][0]  = 0;
        img_batch.logits[i]     = 0;
    }

    if (llama_decode(ctx, img_batch) != 0) {
        fprintf(stderr, "Failed to decode image embeddings\n");
        return 1;
    }
    fprintf(stderr, "Image embeddings injected ✓\n");

    // ── Batch 2: text prompt tokens ───────────────────────────────────────────
    struct llama_batch txt_batch = llama_batch_init(n_prompt_tokens, 0, 1);
    txt_batch.n_tokens = n_prompt_tokens;
    for (int i = 0; i < n_prompt_tokens; i++) {
        txt_batch.token[i]      = prompt_tokens[i];
        txt_batch.pos[i]        = n_img_tokens + i;  // offset by image tokens
        txt_batch.n_seq_id[i]   = 1;
        txt_batch.seq_id[i][0]  = 0;
        txt_batch.logits[i]     = (i == n_prompt_tokens - 1) ? 1 : 0;
    }

    if (llama_decode(ctx, txt_batch) != 0) {
        fprintf(stderr, "Failed to decode text prompt\n");
        return 1;
    }
    fprintf(stderr, "Text prompt processed ✓\nGenerating...\n\n");

    // ── Autoregressive generation ─────────────────────────────────────────────
    llama_token eos = llama_vocab_eos(vocab);
    int n_pos = n_img_tokens + n_prompt_tokens;

    struct llama_sampler* sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    for (int i = 0; i < MAX_NEW_TOKENS; i++) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        if (new_token == eos) break;

        // Print token
        char piece[64] = {0};
        llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);
        printf("%s", piece);
        fflush(stdout);

        // Prepare next batch
        struct llama_batch next = llama_batch_init(1, 0, 1);
        next.n_tokens       = 1;
        next.token[0]       = new_token;
        next.pos[0]         = n_pos++;
        next.n_seq_id[0]    = 1;
        next.seq_id[0][0]   = 0;
        next.logits[0]      = 1;

        llama_sampler_accept(sampler, new_token);

        if (llama_decode(ctx, next) != 0) break;
        llama_batch_free(next);
    }
    printf("\n");

    // ── Cleanup ───────────────────────────────────────────────────────────────
    llama_sampler_free(sampler);
    llama_batch_free(img_batch);
    llama_batch_free(txt_batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(img_embd);
    free(prompt_tokens);

    return 0;
}