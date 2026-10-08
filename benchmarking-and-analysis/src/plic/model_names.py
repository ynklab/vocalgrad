"""Model identifiers shared by inference and CPU-only analysis."""
def slugify_model_id(model_id: str) -> str:
    lowered = model_id.lower()
    if lowered == "moonshotai/kimi-audio-7b-instruct":
        return "kimi-audio"
    if lowered == "nvidia/audio-flamingo-3-hf":
        return "audioflamingo3"
    if lowered == "stepfun-ai/step-audio-2-mini":
        return "step-audio-2-mini"
    if lowered == "xiaomimimo/mimo-audio-7b-instruct":
        return "mimo-audio"
    return model_id.split("/")[-1].lower().replace("/", "-")

