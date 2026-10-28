"""Local invariant checks: python -m unittest reasoning.train.test_sft -v."""
import unittest

from .sft import encode_row, pad_batch


class Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "USER:" + messages[0]["content"] + "\nASSISTANT:"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text]}


class CompletionTests(unittest.TestCase):
    def test_prompt_mask_and_eos(self):
        item = encode_row({"question": "1+1?", "solution": "Add one.", "gold": "2"}, Tokenizer(), 1000)
        labels = [token for token in item["labels"] if token != -100]
        self.assertEqual(labels, [ord(c) for c in "Add one.\n#### 2"] + [0])
        self.assertTrue(all(label == -100 for label in item["labels"][:-len(labels)]))

    def test_answer_only_and_padding_when_pad_equals_eos(self):
        a = encode_row({"question": "1?", "rationale": "", "gold": "1"}, Tokenizer(), 1000)
        b = encode_row({"question": "A longer question?", "solution": "Work", "gold": "3"}, Tokenizer(), 1000)
        batch = pad_batch([a, b], 0)
        end = len(a["input_ids"]) - 1
        self.assertEqual(batch["labels"][0][end], 0)
        self.assertTrue(all(label == -100 for label in batch["labels"][0][end + 1:]))
        self.assertEqual(batch["attention_mask"][0][end], 1)
        self.assertEqual(batch["attention_mask"][0][-1], 0)

    def test_overlength_raises_instead_of_cutting_answer(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            encode_row({"id": "example", "question": "Q", "solution": "Work", "gold": "1"}, Tokenizer(), 1)


if __name__ == "__main__":
    unittest.main()
