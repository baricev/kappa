"""Gemma 3 tokenizer special-token ids (match ``gemma/gm/text/_tokenizer._Gemma3SpecialTokens``).

Do not use ``SentencePieceProcessor.bos_id()`` / ``eos_id()`` as the source of truth for Gemma:
those can differ from the ids the checkpoint and training recipe use.
"""

GEMMA3_PAD = 0
GEMMA3_EOS = 1
GEMMA3_BOS = 2
GEMMA3_UNK = 3
# ``<end_of_turn>`` as a single SentencePiece token (verify with ``tokenizer.model`` if you change tokenizer).
GEMMA3_END_OF_TURN = 106
