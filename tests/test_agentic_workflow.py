from __future__ import annotations

import py_compile
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


FILES_TO_COMPILE = [
    "glayout_agentic/run_agent.py",
    "glayout_agentic/check_env.py",
    "glayout_agentic/progressive_hello_world.py",
    "glayout_agentic/examples/two_fet_shared_diffusion.py",
    "glayout_agentic/examples/two_fet_interdigitized.py",
    "glayout_agentic/examples/two_fet_separate.py",
    "glayout_agentic/skills/two_fet_shared_diffusion_template.py",
    "glayout_agentic/training/train_qwen_lora.py",
    "glayout_agentic/workflow/__init__.py",
    "glayout_agentic/workflow/agent.py",
    "glayout_agentic/workflow/artifacts.py",
    "glayout_agentic/workflow/backends.py",
    "glayout_agentic/workflow/dataset.py",
    "glayout_agentic/workflow/prompts.py",
    "glayout_agentic/workflow/skills.py",
    "glayout_agentic/workflow/validator.py",
]


class AgenticWorkflowTests(unittest.TestCase):
    def test_expected_files_exist(self) -> None:
        required = [
            "glayout_agentic/README.md",
            "glayout_agentic/prompts/system_prompt.txt",
            "glayout_agentic/prompts/repair_prompt.txt",
            "glayout_agentic/skills/two_fet_shared_diffusion.md",
            "glayout_agentic/skills/two_fet_shared_diffusion_template.py",
            "glayout_agentic/examples/two_fet_shared_diffusion.py",
            "glayout_agentic/examples/two_fet_interdigitized.py",
            "glayout_agentic/examples/two_fet_separate.py",
            "glayout_agentic/progressive_hello_world.py",
            "glayout_agentic/training/train_qwen_lora.py",
        ]
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((REPO_ROOT / relative).exists())

    def test_python_files_compile(self) -> None:
        for relative in FILES_TO_COMPILE:
            with self.subTest(path=relative):
                py_compile.compile(
                    str(REPO_ROOT / relative),
                    doraise=True,
                )


if __name__ == "__main__":
    unittest.main()
