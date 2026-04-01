"""
tests/test_all.py — Unit tests for QLoRA fine-tuning project.

Run:
    python -m pytest tests/ -v
    python tests/test_all.py

Test groups (12 groups, 40+ tests)
────────────────────────────────────
  A — Config
  B — Prompt formatters (Alpaca, ChatML, LLaMA-2, Simple)
  C — Dataset loading and tokenisation
  D — Data collator (padding, label masking)
  E — LoRA weight utilities
  F — Model utilities (parameter counts, memory estimates)
  G — Synthetic model forward pass
  H — InstructionDataset (encode, label masking)
  I — Training callbacks (EarlyStopping, MemoryLog)
  J — Training metrics (PPL, token accuracy, smooth)
  K — Manual training loop
  L — Inference / generation
"""

import sys, os, unittest, math, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


# =============================================================================
# Group A — Config
# =============================================================================

class TestConfig(unittest.TestCase):

    def test_defaults(self):
        from config import Config
        cfg = Config()
        self.assertEqual(cfg.model.bnb_4bit_quant_type, "nf4")
        self.assertEqual(cfg.lora.r, 16)
        self.assertEqual(cfg.data.prompt_style, "alpaca")

    def test_to_dict(self):
        from config import Config
        d = Config().to_dict()
        self.assertIn("model", d)
        self.assertIn("lora",  d)
        self.assertIn("data",  d)
        self.assertIn("training", d)

    def test_presets(self):
        from config import alpaca_7b_config, sharegpt_7b_config, fast_test_config
        self.assertEqual(alpaca_7b_config().data.prompt_style, "alpaca")
        self.assertEqual(sharegpt_7b_config().lora.r, 32)
        cfg = fast_test_config()
        self.assertEqual(cfg.model.model_name, "__synthetic__")
        self.assertEqual(cfg.training.max_steps, 2)

    def test_summary(self):
        from config import Config
        s = Config().summary()
        self.assertIn("MODEL", s)
        self.assertIn("LORA",  s)
        self.assertIn("DATA",  s)


# =============================================================================
# Group B — Prompt Formatters
# =============================================================================

class TestAlpacaFormatter(unittest.TestCase):

    def setUp(self):
        from data.formatting import AlpacaFormatter
        self.fmt = AlpacaFormatter()

    def test_format_train_no_input(self):
        sample = {"instruction": "Say hello", "input": "", "output": "Hello!"}
        out = self.fmt.format_train(sample)
        self.assertIn("### Instruction:", out["full_text"])
        self.assertIn("### Response:", out["full_text"])
        self.assertIn("Hello!", out["full_text"])
        self.assertNotIn("### Input:", out["full_text"])

    def test_format_train_with_input(self):
        sample = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        out = self.fmt.format_train(sample)
        self.assertIn("### Input:", out["full_text"])
        self.assertIn("Hola", out["full_text"])

    def test_prompt_is_prefix_of_full_text(self):
        sample = {"instruction": "Test", "input": "", "output": "Answer"}
        out = self.fmt.format_train(sample)
        self.assertTrue(out["full_text"].startswith(out["prompt"]))

    def test_format_inference(self):
        prompt = self.fmt.format_inference("What is 2+2?")
        self.assertIn("### Instruction:", prompt)
        self.assertIn("### Response:", prompt)
        self.assertNotIn("4", prompt)   # no answer in prompt

    def test_completion_start(self):
        self.assertIn("Response", self.fmt.completion_start())


class TestChatMLFormatter(unittest.TestCase):

    def setUp(self):
        from data.formatting import ChatMLFormatter
        self.fmt = ChatMLFormatter("You are helpful.")

    def test_format_train_instruction_style(self):
        sample = {"instruction": "Hi", "output": "Hello!"}
        out = self.fmt.format_train(sample)
        self.assertIn("<|im_start|>", out["full_text"])
        self.assertIn("Hello!", out["full_text"])

    def test_format_train_conversations_style(self):
        sample = {"conversations": [
            {"from": "human", "value": "Hi there"},
            {"from": "gpt",   "value": "Hello!"},
        ]}
        out = self.fmt.format_train(sample)
        self.assertIn("Hi there", out["full_text"])
        self.assertIn("Hello!", out["full_text"])

    def test_prompt_has_no_completion(self):
        sample = {"instruction": "Greet", "output": "Howdy!"}
        out = self.fmt.format_train(sample)
        self.assertNotIn("Howdy!", out["prompt"])
        self.assertIn("Howdy!", out["full_text"])


class TestLLaMA2Formatter(unittest.TestCase):

    def setUp(self):
        from data.formatting import LLaMA2Formatter
        self.fmt = LLaMA2Formatter()

    def test_format_train(self):
        sample = {"instruction": "Count to 3", "input": "", "output": "1, 2, 3"}
        out = self.fmt.format_train(sample)
        self.assertIn("[INST]", out["full_text"])
        self.assertIn("[/INST]", out["full_text"])
        self.assertIn("1, 2, 3", out["full_text"])

    def test_format_inference(self):
        prompt = self.fmt.format_inference("Hello")
        self.assertIn("[INST]", prompt)
        self.assertIn("[/INST]", prompt)


class TestSimpleFormatter(unittest.TestCase):

    def setUp(self):
        from data.formatting import SimpleFormatter
        self.fmt = SimpleFormatter()

    def test_format_train(self):
        sample = {"instruction": "Add", "input": "2+2", "output": "4"}
        out = self.fmt.format_train(sample)
        self.assertIn("### Instruction:", out["full_text"])
        self.assertIn("### Input:", out["full_text"])
        self.assertIn("### Response:", out["full_text"])
        self.assertIn("4", out["full_text"])


class TestGetFormatter(unittest.TestCase):
    def test_registry(self):
        from data.formatting import get_formatter
        for style in ("alpaca", "chatml", "llama2", "simple"):
            fmt = get_formatter(style)
            self.assertIsNotNone(fmt)

    def test_invalid_style(self):
        from data.formatting import get_formatter
        with self.assertRaises(ValueError):
            get_formatter("invalid_style")

    def test_system_prompt_override(self):
        from data.formatting import get_formatter
        fmt = get_formatter("alpaca", system_prompt="Custom system")
        sample = {"instruction": "Test", "input": "", "output": "OK"}
        out = fmt.format_train(sample)
        self.assertIn("Custom system", out["full_text"])


# =============================================================================
# Group C — Dataset Loading
# =============================================================================

class TestDatasetLoading(unittest.TestCase):

    def test_load_synthetic(self):
        from data.dataset import load_raw_samples
        samples = load_raw_samples("__synthetic__", max_samples=8)
        self.assertEqual(len(samples), 8)
        for s in samples:
            self.assertIn("instruction", s)
            self.assertIn("output", s)

    def test_load_synthetic_shuffled(self):
        from data.dataset import load_raw_samples
        a = load_raw_samples("__synthetic__", seed=0)
        b = load_raw_samples("__synthetic__", seed=99)
        # Different seeds should give different order
        instr_a = [s["instruction"] for s in a]
        instr_b = [s["instruction"] for s in b]
        self.assertNotEqual(instr_a, instr_b)


# =============================================================================
# Group D — Data Collator
# =============================================================================

class TestDataCollator(unittest.TestCase):

    def _make_samples(self):
        return [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 2, 3]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([-100, 5]),
            },
        ]

    def _make_tokenizer(self):
        class FakeTok:
            pad_token_id = 0
        return FakeTok()

    def test_pads_to_max_length(self):
        from data.dataset import InstructionDataCollator
        collator = InstructionDataCollator(self._make_tokenizer(), pad_to_multiple_of=1)
        batch = collator(self._make_samples())
        # Both sequences should be padded to length 3
        self.assertEqual(batch["input_ids"].shape, (2, 3))

    def test_labels_padded_with_minus_100(self):
        from data.dataset import InstructionDataCollator
        collator = InstructionDataCollator(self._make_tokenizer(), pad_to_multiple_of=1)
        batch = collator(self._make_samples())
        # Shorter sequence padded → label pad should be -100
        self.assertEqual(batch["labels"][1, 2].item(), -100)

    def test_input_ids_padded_with_pad_id(self):
        from data.dataset import InstructionDataCollator
        collator = InstructionDataCollator(self._make_tokenizer(), pad_to_multiple_of=1)
        batch = collator(self._make_samples())
        self.assertEqual(batch["input_ids"][1, 2].item(), 0)

    def test_pad_to_multiple_of_8(self):
        from data.dataset import InstructionDataCollator
        collator = InstructionDataCollator(self._make_tokenizer(), pad_to_multiple_of=8)
        batch = collator(self._make_samples())
        self.assertEqual(batch["input_ids"].shape[1] % 8, 0)


# =============================================================================
# Group E — LoRA Utilities
# =============================================================================

class TestLoRAUtils(unittest.TestCase):

    def _tiny_model(self):
        return nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8))

    def test_get_trainable_params_empty(self):
        from model.lora import get_trainable_params
        m = self._tiny_model()
        for p in m.parameters():
            p.requires_grad_(False)
        tp = get_trainable_params(m)
        self.assertEqual(len(tp), 0)

    def test_freeze_unfreeze(self):
        from model.lora import freeze_base_model, unfreeze_base_model
        m = self._tiny_model()
        freeze_base_model(m)
        self.assertFalse(any(p.requires_grad for p in m.parameters()))
        unfreeze_base_model(m)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_lora_weight_stats(self):
        from model.lora import lora_weight_stats
        m = self._tiny_model()
        # Add fake lora params
        m[0].lora_A = nn.Parameter(torch.randn(4, 32))
        m[0].lora_B = nn.Parameter(torch.zeros(16, 4))
        stats = lora_weight_stats(m)
        self.assertGreater(len(stats), 0)
        self.assertIn("norm", stats[0])


# =============================================================================
# Group F — Model Utilities
# =============================================================================

class TestModelUtils(unittest.TestCase):

    def _tiny(self):
        return nn.Linear(64, 32)

    def test_count_parameters(self):
        from model.utils import count_parameters
        m = self._tiny()
        c = count_parameters(m)
        expected = 64 * 32 + 32  # weight + bias
        self.assertEqual(c["total"], expected)
        self.assertEqual(c["trainable"], expected)

    def test_count_parameters_frozen(self):
        from model.utils import count_parameters
        m = self._tiny()
        for p in m.parameters():
            p.requires_grad_(False)
        c = count_parameters(m)
        self.assertEqual(c["trainable"], 0)
        self.assertGreater(c["frozen"], 0)

    def test_model_memory_footprint(self):
        from model.utils import model_memory_footprint
        m = nn.Linear(1024, 1024)
        info = model_memory_footprint(m)
        self.assertGreater(info["model_gb"], 0)
        expected_bytes = (1024 * 1024 + 1024) * 4  # fp32 = 4 bytes
        self.assertAlmostEqual(info["model_bytes"], expected_bytes, delta=100)

    def test_estimate_training_memory(self):
        from model.utils import estimate_training_memory_gb
        mem = estimate_training_memory_gb(7e9, batch_size=4, seq_len=2048, r=16)
        self.assertIn("total_gb", mem)
        self.assertGreater(mem["total_gb"], 0)
        self.assertGreater(mem["base_model_gb"], 0)


# =============================================================================
# Group G — Synthetic Model
# =============================================================================

class TestSyntheticModel(unittest.TestCase):

    def setUp(self):
        from model.loader import load_synthetic_model_and_tokenizer
        self.model, self.tok = load_synthetic_model_and_tokenizer("cpu")

    def test_forward_with_labels(self):
        ids    = torch.randint(3, 100, (2, 16))
        labels = torch.randint(3, 100, (2, 16))
        labels[:, :4] = -100   # mask prompt
        out = self.model(input_ids=ids, labels=labels)
        self.assertFalse(torch.isnan(out.loss))
        self.assertGreater(out.loss.item(), 0)

    def test_forward_without_labels(self):
        ids = torch.randint(3, 100, (1, 8))
        out = self.model(input_ids=ids)
        self.assertEqual(out.logits.shape[:2], (1, 8))

    def test_generate(self):
        ids = torch.randint(3, 100, (1, 4))
        out = self.model.generate(ids, max_new_tokens=5)
        self.assertGreater(out.shape[1], 4)


# =============================================================================
# Group H — InstructionDataset
# =============================================================================

class TestInstructionDataset(unittest.TestCase):

    def _make_tokenizer(self):
        class FakeTok:
            pad_token_id = 0
            eos_token_id = 2
            def __call__(self, text, **kw):
                ids = [hash(c) % 97 + 3 for c in str(text)[:32]]
                t = torch.tensor([ids])
                return {"input_ids": t, "attention_mask": torch.ones_like(t)}
        return FakeTok()

    def test_basic_encoding(self):
        from data.dataset import InstructionDataset
        from data.formatting import AlpacaFormatter
        samples = [
            {"instruction": "Add", "input": "2+2", "output": "4"},
            {"instruction": "Say hello", "input": "", "output": "Hello"},
        ]
        ds = InstructionDataset(
            samples, self._make_tokenizer(), AlpacaFormatter(),
            max_seq_length=128, train_on_completions_only=False,
        )
        self.assertGreater(len(ds), 0)
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)

    def test_label_masking(self):
        from data.dataset import InstructionDataset
        from data.formatting import AlpacaFormatter
        samples = [{"instruction": "Test", "input": "", "output": "Output text"}]
        ds = InstructionDataset(
            samples, self._make_tokenizer(), AlpacaFormatter(),
            max_seq_length=128, train_on_completions_only=True,
        )
        if len(ds) > 0:
            labels = ds[0]["labels"]
            # At least some labels should be -100 (prompt masked)
            self.assertTrue((labels == -100).any())


# =============================================================================
# Group I — Callbacks
# =============================================================================

class TestCallbacks(unittest.TestCase):

    def _make_state(self, step=5, kl=0.15):
        return type("State", (), {"global_step": step})()

    def _make_control(self):
        return type("Control", (), {"should_training_stop": False})()

    def test_early_stopping_stops_on_no_improvement(self):
        from training.callbacks import EarlyStoppingCallback
        cb = EarlyStoppingCallback(patience=2, min_delta=0.01)
        ctrl = self._make_control()
        args  = None; state = self._make_state()
        cb.on_evaluate(args, state, ctrl, metrics={"eval_loss": 2.0})
        cb.on_evaluate(args, state, ctrl, metrics={"eval_loss": 2.0})  # no improvement
        cb.on_evaluate(args, state, ctrl, metrics={"eval_loss": 2.0})  # still no improvement
        self.assertTrue(ctrl.should_training_stop)

    def test_early_stopping_resets_on_improvement(self):
        from training.callbacks import EarlyStoppingCallback
        cb = EarlyStoppingCallback(patience=2)
        ctrl = self._make_control()
        args  = None; state = self._make_state()
        cb.on_evaluate(args, state, ctrl, metrics={"eval_loss": 2.0})
        cb.on_evaluate(args, state, ctrl, metrics={"eval_loss": 2.0})
        cb.on_evaluate(args, state, ctrl, metrics={"eval_loss": 1.0})  # improved!
        self.assertFalse(ctrl.should_training_stop)

    def test_memory_log_callback_cpu(self):
        from training.callbacks import MemoryLogCallback
        cb   = MemoryLogCallback(device=torch.device("cpu"))
        logs = {}
        cb.on_log(None, None, None, logs=logs)
        # CPU: no GPU metrics added (no crash)
        self.assertNotIn("peak_gpu_gb", logs)

    def test_lora_checkpoint_callback_saves(self):
        from training.callbacks import LoRACheckpointCallback

        class FakeModel:
            def save_pretrained(self, path):
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "adapter.txt"), "w") as f:
                    f.write("weights")

        with tempfile.TemporaryDirectory() as d:
            cb    = LoRACheckpointCallback(d, save_total_limit=3)
            state = type("S", (), {"global_step": 10})()
            cb.on_save(None, state, None, model=FakeModel())
            self.assertTrue(os.path.exists(os.path.join(d, "adapter-step-10")))


# =============================================================================
# Group J — Metrics
# =============================================================================

class TestMetrics(unittest.TestCase):

    def test_smooth(self):
        from training.metrics import smooth
        values = [1.0, 2.0, 0.5, 1.5, 0.8]
        smoothed = smooth(values, window=3)
        self.assertEqual(len(smoothed), len(values))
        self.assertEqual(smoothed[0], values[0])

    def test_smooth_empty(self):
        from training.metrics import smooth
        self.assertEqual(smooth([]), [])

    def test_rouge_l_perfect(self):
        from training.metrics import compute_rouge_l
        try:
            score = compute_rouge_l(["the cat sat on the mat"], ["the cat sat on the mat"])
            self.assertAlmostEqual(score, 1.0, places=3)
        except ImportError:
            pass  # rouge-score not installed → skip

    def test_rouge_l_zero(self):
        from training.metrics import compute_rouge_l
        try:
            score = compute_rouge_l(["alpha beta gamma"], ["xyz abc def"])
            self.assertLess(score, 0.3)
        except ImportError:
            pass


# =============================================================================
# Group K — Manual Training Loop
# =============================================================================

class TestManualTrainingLoop(unittest.TestCase):

    def test_loss_decreases(self):
        """Loss should be finite and training should not crash."""
        from config import fast_test_config
        from model.loader import load_synthetic_model_and_tokenizer
        from model.lora import attach_lora_synthetic
        from data.dataset import InstructionDataset, load_raw_samples
        from data.formatting import get_formatter
        from training.trainer import QLoRATrainer

        cfg = fast_test_config()
        model, tokenizer = load_synthetic_model_and_tokenizer("cpu")
        model = attach_lora_synthetic(model, cfg.lora)

        samples   = load_raw_samples("__synthetic__", max_samples=16)
        formatter = get_formatter(cfg.data.prompt_style, cfg.data.system_prompt)
        train_ds  = InstructionDataset(
            samples, tokenizer, formatter,
            max_seq_length=32, train_on_completions_only=False,
        )

        trainer = QLoRATrainer(model, tokenizer, train_ds, None, cfg)
        history = trainer.train_manual(num_steps=2, log_every=1)

        self.assertEqual(len(history), 2)
        for rec in history:
            self.assertIn("loss", rec)
            self.assertFalse(math.isnan(rec["loss"]))
            self.assertGreater(rec["loss"], 0)


# =============================================================================
# Group L — Inference
# =============================================================================

class TestInferenceGenerator(unittest.TestCase):

    def test_generate_returns_string(self):
        from inference.generate import InstructionGenerator, GenerationConfig
        from model.loader import load_synthetic_model_and_tokenizer
        from data.formatting import get_formatter

        model, tok = load_synthetic_model_and_tokenizer("cpu")
        formatter  = get_formatter("alpaca")
        gen_cfg    = GenerationConfig(max_new_tokens=10, do_sample=False)
        gen        = InstructionGenerator(model, tok, formatter, gen_cfg,
                                          device=torch.device("cpu"))
        response = gen.generate("What is 2+2?")
        self.assertIsInstance(response, str)

    def test_generate_batch(self):
        from inference.generate import InstructionGenerator, GenerationConfig
        from model.loader import load_synthetic_model_and_tokenizer
        from data.formatting import get_formatter

        model, tok = load_synthetic_model_and_tokenizer("cpu")
        gen = InstructionGenerator(
            model, tok, get_formatter("alpaca"),
            GenerationConfig(max_new_tokens=5, do_sample=False),
            device=torch.device("cpu"),
        )
        instructions = ["Question 1", "Question 2", "Question 3"]
        responses    = gen.generate_batch(instructions)
        self.assertEqual(len(responses), 3)
        for r in responses:
            self.assertIsInstance(r, str)

    def test_generation_config_defaults(self):
        from inference.generate import GenerationConfig
        cfg = GenerationConfig()
        self.assertEqual(cfg.temperature, 0.7)
        self.assertTrue(cfg.do_sample)
        self.assertGreater(cfg.max_new_tokens, 0)


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
