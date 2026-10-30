import json
import tempfile
import unittest
from pathlib import Path

from .run_final import attach_validation, verify_adapter, verify_inputs
from .sft import read_rows


class FinalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path('configs/sft-final.json').read_text())

    def test_frozen_inputs_match_and_changed_data_is_rejected(self):
        verify_inputs(self.protocol)
        changed = {**self.protocol, 'data_sha256': {**self.protocol['data_sha256'], 'long': 'wrong'}}
        with self.assertRaisesRegex(ValueError, 'dataset changed'):
            verify_inputs(changed)

    def test_pilot_reuse_requires_config_hashes_and_adapter(self):
        p = self.protocol
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = {'status': 'complete', 'n_train': 6547, 'n_val': 500,
                   'config': {'model': 'Qwen/Qwen2.5-0.5B', 'train_file': 'data/sft/standard.jsonl',
                              'seed': 0, 'limit': None, 'val_limit': None, 'learning_rate': 0.0001,
                              'rank': 8, 'epochs': 1, 'precision': 'float32', 'batch_size': 1,
                              'grad_accum': 8, 'max_length': 4096, 'max_new_tokens': 512, 'val_batch_size': 1},
                   'data_sha256': {'train': p['data_sha256']['standard'], 'validation': p['validation_sha256']}}
            (root / 'train_log.json').write_text(json.dumps(log))
            (root / 'adapter_model.safetensors').touch()
            lora = {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05, 'bias': 'none',
                    'target_modules': p['target_modules']}
            (root / 'adapter_config.json').write_text(json.dumps(lora))
            verify_adapter(root, p, 'standard', '05B', 0)
            with self.assertRaisesRegex(ValueError, 'seed'):
                verify_adapter(root, p, 'standard', '05B', 1)
            log['config']['limit'] = 64
            (root / 'train_log.json').write_text(json.dumps(log))
            with self.assertRaisesRegex(ValueError, 'limit'):
                verify_adapter(root, p, 'standard', '05B', 0)

    def test_attach_preserves_hf_score_and_rejects_wrong_question_order(self):
        rows = read_rows('data/splits/gsm8k_val.jsonl')
        raw = [{**r, 'samples': [{'text': '#### ' + r['gold']}]} for r in rows]
        scored = [{'id': r['id'], 'correct': [True]} for r in rows]
        summary = {'n_questions': 500, 'n_samples': 1, 'sample_acc': 1.0, 'wall_seconds': 10}
        log = {'status': 'complete', 'epochs': [{'epoch': 1, 'gsm8k_val_accuracy': 0.602}]}
        updated = attach_validation(log, raw, scored, summary, self.protocol, Path('runs/mock.jsonl'))
        self.assertEqual(updated['epochs'][0]['hf_gsm8k_val_accuracy'], 0.602)
        self.assertEqual(updated['epochs'][0]['gsm8k_val_accuracy'], 1.0)
        self.assertEqual(updated['epochs'][0]['validation_backend'], 'vllm')
        with self.assertRaisesRegex(ValueError, 'IDs/gold/order'):
            attach_validation(log, raw[::-1], scored, summary, self.protocol, Path('runs/mock.jsonl'))
        with self.assertRaisesRegex(ValueError, 'Summary accuracy'):
            attach_validation(log, raw, scored, {**summary, 'sample_acc': 0.5}, self.protocol, Path('runs/mock.jsonl'))


if __name__ == '__main__':
    unittest.main()
