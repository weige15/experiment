from m1_analysis import content_indices


class TinyTokenizer:
    all_special_ids = [9]
    vocabulary = {0: ' yes', 1: '\n\n', 2: 'Q', 3: ':', 4: 'reason', 5: 'ing', 6: ' ', 9: ''}
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.vocabulary[i] for i in ids)


def test_content_mask_excludes_delimiter_and_eos_not_the_answer():
    tokenizer = TinyTokenizer()
    assert content_indices(tokenizer, [0, 1, 2, 3]) == [0]
    assert content_indices(tokenizer, [0, 9]) == [0]
    assert content_indices(tokenizer, [4, 5]) == [0, 1]
    assert content_indices(tokenizer, [4, 5, 6]) == [0, 1]
    assert content_indices(tokenizer, [9]) == []
