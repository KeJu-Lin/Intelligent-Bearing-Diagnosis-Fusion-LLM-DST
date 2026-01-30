#!/usr/bin/env python3
import sys
import warnings
import importlib


# Collect warnings emitted during model load
warnings_list = []

def warn_capture(message, category, filename, lineno, file=None, line=None):
    warnings_list.append(str(message))


# Override the global warning handler
warnings.showwarning = warn_capture


try:
    # Avoid triggering `transformers.__init__` by importing ONLY the submodule.
    # This module always contains AutoModelForCausalLM.
    module = importlib.import_module(
        "transformers.models.auto.modeling_auto"
    )
    AutoModelForCausalLM = module.AutoModelForCausalLM

    # AutoTokenizer is needed too
    tokenizer_module = importlib.import_module(
        "transformers.models.auto.tokenization_auto"
    )
    AutoTokenizer = tokenizer_module.AutoTokenizer

    # Try loading the local model
    tok = AutoTokenizer.from_pretrained("./MODEL")
    model = AutoModelForCausalLM.from_pretrained("./MODEL")

except Exception as e:
    print("EXCEPTION:", e)
    # 125 meaning “skip this commit”
    sys.exit(125)


# Check results
if warnings_list:
    print("BAD: warning triggered")
    sys.exit(1)

print("GOOD: no warnings")
sys.exit(0)
