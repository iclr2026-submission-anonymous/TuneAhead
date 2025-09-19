def select_feature_groups(X_all):
    names = X_all.columns.tolist()
    def pick(keys):
        return sorted({f for f in names if any(k in f for k in keys)})

    static = pick([
        "avg_input_", "avg_output_", "ttr", "semantic_", "approximate_duplicates",
        "ngram", "consistency", "base_model_perplexity", "tfidf", "sbert", "jsd", "vocab_size"
    ])
    dynamic = pick([
        "loss", "grad", "param_change", "activation_sparsity", "initial_loss",
        "landscape_flatness"
    ])
    hyper = [c for c in names if c in ("learning_rate","batch_size")]

    # 确保至少都有超参
    return sorted(set(static + hyper)), sorted(set(dynamic + hyper))
