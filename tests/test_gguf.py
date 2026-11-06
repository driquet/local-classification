"""llama.cpp serving: its torch-free chat rendering must give the token ids transformers gives."""

import pytest

from lc.models.decision import messages, request

pytest.importorskip("llama_cpp")
QWEN = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
ONEJEV = ("OmniJev/OneJev1.6-4B", "bb248003b6eb771a995cb3672ad47db090a9cd48")


@pytest.mark.parametrize("model", [QWEN, ONEJEV])
def test_chat_ids_match_transformers(model):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from lc.models.gguf import Chat

    hf, chat = AutoTokenizer.from_pretrained(snapshot_download(model[0], revision=model[1])), Chat(*model)
    instructions, options, _ = request(
        "dynasent", 'Café\'s <b>great</b> "😀"\n  ok', ["positive", "negative"]
    )
    for conversation in (
        messages(instructions, options),
        [{"role": "user", "content": "Intent: pto request"}],
    ):
        text = hf.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert chat.prompt_ids(conversation) == hf.encode(text, add_special_tokens=False)
