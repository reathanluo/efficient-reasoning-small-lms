"""Local invariant checks: python -m unittest reasoning.train.test_sft -v."""
import unittest
import json
import tempfile
from pathlib import Path

from .sft import check_training_metrics, encode_row, load_resume_log, pad_batch, validate_rows
from .check_validation import compare
from .prepare_long_probe import select_indices


class Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "USER:" + messages[0]["content"] + "\nASSISTANT:"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text]}


class CompletionTests(unittest.TestCase):
    def test_resume_checks_inputs_config_and_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            checkpoint = out / "checkpoint-4"
            checkpoint.mkdir()
            config = {"precision": "float32", "epochs": 1}
            hashes = {"train": "train-hash", "validation": "val-hash"}
            log = {"config": config, "data_sha256": hashes, "status": "failed",
                   "epochs": [{"epoch": 1, "global_step": 8}], "error": "old error"}
            (out / "train_log.json").write_text(json.dumps(log))
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json",
                         "adapter_model.safetensors"):
                (checkpoint / name).touch()
            (checkpoint / "trainer_state.json").write_text('{"global_step": 4}')
            resumed = load_resume_log(out, checkpoint, config, hashes)
            self.assertEqual(resumed["status"], "running")
            self.assertEqual(resumed["epochs"], [])
            self.assertNotIn("error", resumed)
            self.assertEqual(resumed["resume_events"][-1]["global_step"], 4)
            with self.assertRaisesRegex(ValueError, "changed epochs"):
                load_resume_log(out, checkpoint, {**config, "epochs": 2}, hashes)
            with self.assertRaisesRegex(ValueError, "hashes"):
                load_resume_log(out, checkpoint, config, {"train": "different"})
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                load_resume_log(out, checkpoint, config, hashes)

    def test_long_probe_contains_actual_max_and_eight_distinct_examples(self):
        lengths = [20, 1000, 30, 4039] + list(range(100, 200))
        indices = select_indices(lengths)
        self.assertEqual(indices[0], 3)
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(set(indices)), 8)
        self.assertEqual(min(lengths[i] for i in indices), 20)

    def test_invalid_training_metrics_are_rejected(self):
        for metrics in ({"loss": float("nan")}, {"grad_norm": float("inf")},
                        {"loss": 0.0, "grad_norm": float("nan")}):
            with self.assertRaises(FloatingPointError):
                check_training_metrics(metrics)
        check_training_metrics({"loss": 0.5, "grad_norm": 1.2})
        check_training_metrics({"train_runtime": 10})

    def test_comparison_distinguishes_wording_answer_and_correctness(self):
        first = [{"id": "a", "text": "#### 2", "correct": True},
                 {"id": "b", "text": "#### 3", "correct": False}]
        second = [{"id": "a", "text": "Work\n#### 2", "correct": True},
                  {"id": "b", "text": "#### 4", "correct": True}]
        result = compare(first, second)
        self.assertEqual(result["text_disagreements"], 2)
        self.assertEqual(result["answer_disagreements"], 1)
        self.assertEqual(result["correctness_disagreement_ids"], ["b"])
        with self.assertRaises(ValueError):
            compare(first, second[::-1])

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
