"""Local invariant checks: python -m unittest reasoning.train.test_sft -v."""
import unittest

from .sft import encode_row, pad_batch, validate_rows


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


class ValidationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Validation tests require torch; masking tests do not")
        self.torch = torch
        self.rows = [{"id": str(i), "question": "Q" * (i + 1), "gold": str(i + 1)}
                     for i in range(3)]

        class Encoding(dict):
            def to(self, device):
                return self

        class BatchTokenizer(Tokenizer):
            padding_side = "right"
            pad_token_id = 0

            def __call__(inner, prompts, **kwargs):
                assert inner.padding_side == "left"
                ids = [[ord(c) for c in prompt] for prompt in prompts]
                width = max(map(len, ids))
                return Encoding(
                    input_ids=torch.tensor([[0] * (width - len(seq)) + seq for seq in ids]),
                    attention_mask=torch.tensor([[0] * (width - len(seq)) + [1] * len(seq) for seq in ids]),
                )

            def decode(inner, ids, **kwargs):
                return "".join(chr(i) for i in ids if i != 0)

        class Model:
            training = True
            device = "cpu"

            def __init__(inner):
                inner.seen = 0

            def eval(inner):
                inner.training = False

            def train(inner, mode):
                inner.training = mode

            def generate(inner, input_ids, attention_mask, **kwargs):
                assert kwargs["use_cache"] is True
                assert kwargs["do_sample"] is False
                assert inner.training is False
                completions = []
                for _ in input_ids:
                    inner.seen += 1
                    # Fake an early EOS followed by padding (same ID).
                    completions.append([ord(c) for c in f"#### {inner.seen}"] + [0, 0, 0])
                return torch.cat([input_ids, torch.tensor(completions)], dim=1)

        self.tokenizer = BatchTokenizer()
        self.model = Model()

    def test_batches_preserve_order_trim_padding_and_restore_state(self):
        progress = []
        predictions = validate_rows(self.model, self.tokenizer, self.rows, 2, 10,
                                    lambda done, total, seconds: progress.append((done, total)))
        self.assertEqual([row["id"] for row in predictions], ["0", "1", "2"])
        self.assertTrue(all(row["correct"] for row in predictions))
        self.assertEqual([row["gen_tokens"] for row in predictions], [7, 7, 7])
        self.assertTrue(all(row["finish_reason"] == "stop" for row in predictions))
        self.assertEqual(progress, [(2, 3), (3, 3)])
        self.assertTrue(self.model.training)
        self.assertEqual(self.tokenizer.padding_side, "right")

    def test_batch_one_and_batch_two_match(self):
        first = validate_rows(self.model, self.tokenizer, self.rows, 1, 10)
        self.model.seen = 0
        second = validate_rows(self.model, self.tokenizer, self.rows, 2, 10)
        self.assertEqual(first, second)

    def test_generation_failure_restores_state(self):
        def fail(**kwargs):
            raise RuntimeError("simulated generation failure")
        self.model.generate = fail
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            validate_rows(self.model, self.tokenizer, self.rows, 2, 10)
        self.assertTrue(self.model.training)
        self.assertEqual(self.tokenizer.padding_side, "right")


if __name__ == "__main__":
    unittest.main()
