# Lab Instruction: Modern Python ML Engineering (uv, Project Layouts, and Design Patterns)

Welcome to the Python Machine Learning Engineering Lab. In this lab, you will learn how to transition from a single-file prototype (`main.py`) to a production-grade, modular machine learning project — step by step.

The lab is structured in **three progressive phases**. Each phase builds directly on the previous one, so you can see the project gradually evolve from a messy script into a well-engineered codebase.

---

## Project Baseline Overview

This baseline project implements a deep learning pipeline using a custom VGG-16 architecture to recognize and classify Bangladeshi banknotes (denominations: 1, 2, 5, 10, 20, 50, 100, 500, 1000).

### Hardware & OS Compatibility
This codebase is designed for high portability and cross-platform compatibility, automatically selecting the optimal hardware acceleration backend:
- **NVIDIA GPU (CUDA)**: Automatically used if CUDA drivers and a compatible GPU are detected.
- **AMD GPU (ROCm)**: Supported on Linux via ROCm-enabled PyTorch (maps CUDA calls to AMD HIP backend automatically).
- **Apple Silicon**: Automatically utilizes Metal Performance Shaders (MPS) for local GPU acceleration.
- **CPU Mode**: Safely falls back to CPU execution if no GPU backend or drivers are available.
- **Cross-OS Support**: Optimized for Windows, Linux, and macOS. The multiprocessing startup method is dynamically adjusted to prevent errors and crashes on Windows.

---

## Lab Overview

| Phase | Learning Goal | Difficulty |
| :---: | :--- | :---: |
| Phase 1 | Environment setup with `uv`, run the baseline project | ★☆☆ |
| Phase 2 | Split modules, understand Flat Layout vs. Src Layout | ★★☆ |
| Phase 3 | Refactor with Design Patterns + Code Quality tools | ★★★ |

---

## Phase 1: Environment & Dependency Management with `uv`

**Goal**: Set up a modern Python development environment using `uv` and successfully run the baseline `main.py`.

In modern Python engineering, standard `pip` and virtualenv can be slow and hard to keep reproducible. We will use `uv`, an extremely fast Python packaging tool written in Rust.

### Step 1.1: Install `uv`
Install `uv` globally on your machine:
- **Linux/macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Alternatively (via `pip`)**:
  ```bash
  pip install uv
  ```

### Step 1.2: Initialize the Project and Manage Dependencies
We will use a `pyproject.toml` file to declare dependencies rather than a traditional `requirements.txt`:
1. Run `uv init` in the root of the project to create `pyproject.toml`.
2. Add project dependencies using `uv add` (e.g. `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`).
3. Add development dependencies (linters and type checkers) using `uv add --dev` (e.g. `ruff`, `mypy`).
4. Run `uv sync` to generate the `uv.lock` file and create the `.venv` virtual environment.
5. Execute python scripts using `uv run python main.py`.

> **Key Concept: `pyproject.toml` vs. `uv.lock`**
> - `pyproject.toml` declares **abstract dependencies** (e.g. `torch>=2.1.0`) — it says *what* your project needs.
> - `uv.lock` records **exact resolved versions** (e.g. `torch==2.12.1`) — it locks down *precisely which version* everyone uses, ensuring reproducibility across machines.

### Step 1.3: Run the Baseline Project
After setting up the environment, verify everything works:
```bash
uv run python main.py
```
You should see the training process begin. Confirm that the output shows `Using device: ...` and starts training epochs. You can stop it with `Ctrl+C` after confirming it works.

**✅ Checkpoint**: You should now have a working `pyproject.toml`, a `uv.lock` file, and be able to run `main.py` via `uv run`.

---

## Phase 2: Project Layouts — From Flat to Src

**Goal**: Split the monolithic `main.py` into separate modules, experience both Flat Layout and Src Layout, and understand why Src Layout is the production standard.

### Step 2.1: Identify Code to Extract

Before moving files around, identify three logical groups inside `main.py`:
- **Model definition**: The `VGG16` class (neural network architecture)
- **Dataset utilities**: `CustomDataset`, `CustomTestDataset`, `get_img_info`, `denormalize_image`
- **Training pipeline**: `Config`, `set_seed`, `train_epoch`, `validate`, plotting utilities, and the `main()` function

### Step 2.2: Implement Flat Layout

Create two new files at the project root and move the corresponding code:

```
pypractice-exercise/
├── main.py          ← keeps Config, training loop, main()
├── models.py        ← VGG16 class
├── dataset.py       ← CustomDataset, CustomTestDataset, get_img_info, denormalize_image
├── pyproject.toml
├── uv.lock
└── data/
```

In Flat Layout, modules can be imported directly:
```python
# In main.py
from models import VGG16
from dataset import CustomDataset, CustomTestDataset, get_img_info
```

**✅ Checkpoint**: Run `uv run python main.py` and verify the training starts successfully.

### Step 2.3: Migrate to Src Layout (Production Standard)

The Flat Layout works, but it has a hidden problem: **accidental imports**. Since Python adds the current working directory to `sys.path`, any `.py` file in the root can be imported as a module — even if you didn't intend for it to be part of your package. This can cause subtle bugs in larger projects.

**Src Layout** prevents this by placing all source code under a `src/` directory. Reorganize your project:

```
pypractice-exercise/
├── main.py               ← entry point, stays in root
├── src/
│   └── banknote_classifier/
│       ├── __init__.py
│       ├── models.py
│       └── dataset.py
├── pyproject.toml
├── uv.lock
└── data/
```

> **Why keep `main.py` at the root?**  
> `main.py` serves as the entry point and imports from the `banknote_classifier` package (e.g. `from banknote_classifier.models import VGG16`). When the package is **not installed**, Python cannot find `banknote_classifier` and will throw a `ModuleNotFoundError`. This is by design — it forces you to properly install your package.

### Step 2.4: Configure `pyproject.toml` for Packaging

To make the Src Layout work, you need to tell the build system where your source code lives.

1. Update `pyproject.toml` with the build system configuration:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "banknote-classifier"
   version = "0.1.0"
   description = "Bangla Banknote Classifier"
   readme = "README.md"
   requires-python = ">=3.12"
   dependencies = [
       # TODO: List all your production packages (torch, torchvision, etc.)
   ]
   ```
   > **Hint**: The `[build-system]` section tells `uv` (and `pip`) to use Hatchling as the build backend. Hatchling follows the `src/` layout convention by default, meaning it knows to look for packages under `src/`.

2. **Try running before installing** — observe the error:
   To observe the `ModuleNotFoundError` correctly, run python using the virtual environment interpreter directly without `uv run` (since `uv run` will automatically build and install the package if `pyproject.toml` is present):
   ```bash
   .venv/bin/python main.py
   # Expected: ModuleNotFoundError: No module named 'banknote_classifier'
   ```
   (Alternatively, you can run `uv run --no-project python main.py` to achieve the same effect.)
   This error proves that Src Layout successfully isolates your package — it cannot be imported without proper installation.

3. **Install in editable mode** to fix the import:
   ```bash
   uv pip install -e .
   ```
   > **What does `-e .` do?**  
   > Editable install (`-e`) creates a link from the virtual environment's `site-packages` directory to your `src/` folder. This means your package becomes importable like any installed library, but any code changes you make take effect immediately without reinstalling.

4. Run again and verify:
   ```bash
   uv run python main.py
   # Should now work!
   ```

**✅ Checkpoint**: You should understand why `ModuleNotFoundError` occurs in Src Layout, and how `uv pip install -e .` resolves it.

---

## Phase 3: Design Patterns & Code Quality

**Goal**: Refactor your code using the Registry Pattern and Strategy Pattern to make it extensible, then enforce code quality with `ruff` and `mypy`.

### Task 3A: Implement the Registry Pattern

**What problem does it solve?**  
In large ML projects, you often want to select a model architecture from a config file (e.g. `model_name = "vgg16"`). Without a registry, you'd need ugly `if-elif` chains every time you add a new model. The Registry Pattern lets you map string names to classes automatically.

**Step 1**: Start with a simple dictionary-based registry.

Create a new file `src/banknote_classifier/registry.py`:
```python
import torch.nn as nn
from typing import Dict, Type

# A simple dictionary that maps string names to model classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
```

Then, in `models.py`, manually register your model after defining it:
```python
from .registry import MODEL_REGISTRY

class VGG16(nn.Module):
    # ... (your existing model code)
    pass

# Manually register the model
MODEL_REGISTRY["vgg16"] = VGG16
```

In `main.py`, retrieve the model dynamically. Note that because `main.py` no longer directly references the `VGG16` class, Python will *not* import `models.py` by default. This means the registration code (`MODEL_REGISTRY["vgg16"] = VGG16`) will not be executed, and accessing `MODEL_REGISTRY["vgg16"]` will throw a `KeyError`.

To trigger the registration, you must also import the `models` module in `main.py` for its side-effects:
```python
from banknote_classifier import models  # Triggers registration
from banknote_classifier.registry import MODEL_REGISTRY

# Instead of: model = VGG16(num_classes=9)
model_cls = MODEL_REGISTRY["vgg16"]
model = model_cls(num_classes=Config.NUM_CLASSES).to(device)
```

**✅ Checkpoint**: The training should work exactly as before, but now the model is loaded dynamically via a string key.

**Step 2 (Upgrade)**: Refactor the dictionary into a class with a decorator-based registration system.

The manual approach (`MODEL_REGISTRY["vgg16"] = VGG16`) is fragile — you might forget to register a model. A decorator can automate this.

Refactor `registry.py` into a class:
```python
import torch.nn as nn
from typing import Callable, Dict, Type


class ModelRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        # TODO: Return a decorator function that:
        #   1. Stores the class in self._registry under the given name
        #   2. Returns the class unchanged
        pass

    def get(self, name: str) -> Type[nn.Module]:
        # TODO: Look up `name` in self._registry.
        #   - If found, return the class.
        #   - If not found, raise a ValueError with a helpful message.
        pass


MODEL_REGISTRY = ModelRegistry()
```

Now register the model with a decorator in `models.py`:
```python
from .registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("vgg16")
class VGG16(nn.Module):
    # ... (your existing model code)
    pass
```

And update `main.py` to use the `.get()` method:
```python
model_cls = MODEL_REGISTRY.get("vgg16")
model = model_cls(num_classes=Config.NUM_CLASSES).to(device)
```

> **Understanding the Decorator**  
> `@MODEL_REGISTRY.register("vgg16")` is syntactic sugar. When Python loads `models.py`, it:  
> 1. Calls `MODEL_REGISTRY.register("vgg16")`, which returns a decorator function.  
> 2. Passes the `VGG16` class to that decorator function.  
> 3. The decorator stores `VGG16` in the internal dictionary under the key `"vgg16"`, then returns `VGG16` unchanged.
>
> **Important**: Decorators only run when the module is loaded. If `models.py` is never imported, the registration never happens. Make sure your `__init__.py` imports the models module:
> ```python
> # src/banknote_classifier/__init__.py
> from .models import VGG16 as VGG16  # Use 'as VGG16' to prevent Ruff F401 unused import error
> ```

**✅ Checkpoint**: Training works, and the model is loaded via `MODEL_REGISTRY.get("vgg16")`.

---

### Task 3B: Implement the Strategy Pattern

**What problem does it solve?**  
Your training pipeline uses different image preprocessing for training (with augmentation) vs. validation (without augmentation). Currently, these transform pipelines are hardcoded inside `main()`. The Strategy Pattern encapsulates each preprocessing pipeline into its own class, so you can swap strategies at runtime without modifying the training loop.

**Step 1**: Define the abstract base class and concrete strategies in `src/banknote_classifier/dataset.py`:

```python
from abc import ABC, abstractmethod
from torchvision import transforms


class PreprocessStrategy(ABC):
    """Base class for all preprocessing strategies."""

    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """Return a composed transform pipeline."""
        pass


class StandardPreprocess(PreprocessStrategy):
    """Basic preprocessing: resize, convert to tensor, normalize.
    Suitable for validation and testing."""

    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


class HeavyAugmentation(PreprocessStrategy):
    """Augmented preprocessing: adds random flips and rotations.
    Suitable for training to reduce overfitting."""

    def get_transforms(self) -> transforms.Compose:
        # TODO: Implement a Compose pipeline that includes:
        #   - Resize to 224x224
        #   - RandomHorizontalFlip
        #   - RandomRotation(15)
        #   - ToTensor
        #   - Normalize with ImageNet mean/std
        pass
```

**Step 2**: Update your `CustomDataset` class to accept a `PreprocessStrategy` object.

The key change is that the dataset constructor should be able to accept *either* a raw `transforms.Compose` (for backward compatibility) *or* a `PreprocessStrategy` object:
```python
class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        # If a strategy object is passed, extract its transforms
        if isinstance(transform, PreprocessStrategy):
            self.transform = transform.get_transforms()
        else:
            self.transform = transform
```

**Step 3**: Use the strategies in `main.py`:
```python
from banknote_classifier.dataset import StandardPreprocess, HeavyAugmentation

# Select preprocessing strategies
train_strategy = HeavyAugmentation()
valid_strategy = StandardPreprocess()

# Pass strategies directly to datasets
train_set = CustomDataset(train_imgs, train_lbls, transform=train_strategy)
valid_set = CustomDataset(val_imgs, val_lbls, transform=valid_strategy)

# Don't forget to update the test set as well later in the file!
# test_set = CustomTestDataset(test_paths, transform=valid_strategy)
```

> **Why is this better?**  
> Adding a new strategy (e.g. `MixUpAugmentation`) only requires creating a new subclass of `PreprocessStrategy`. You never need to touch `CustomDataset` or the training loop — just swap the strategy object at the call site.

**✅ Checkpoint**: Training works with strategies. Try switching `train_strategy` to `StandardPreprocess()` and observe the difference.

---

### Task 3C: Code Linting & Formatting with `ruff`

`ruff` is an extremely fast Python linter and formatter written in Rust. It replaces tools like `flake8`, `black`, and `isort` in a single command.

1. Add Ruff configuration to your `pyproject.toml`:
   ```toml
   [tool.ruff]
   line-length = 135

   [tool.ruff.lint]
   select = ["E", "F", "I"]
   ```
   > **What do these rules mean?**
   > - `E`: Style errors (e.g. whitespace issues, line length)
   > - `F`: Logical errors (e.g. unused imports, undefined names)
   > - `I`: Import sorting (groups stdlib, third-party, and local imports)

2. Run the linter and fix issues:
   ```bash
   uv run ruff check src/ main.py        # Find problems
   uv run ruff check src/ main.py --fix  # Auto-fix what it can
   uv run ruff format src/ main.py       # Apply consistent formatting
   ```

**✅ Checkpoint**: `uv run ruff check src/ main.py` reports `All checks passed!`

> 💡 **Matplotlib Import Order and E402 Warning**:
> In `main.py`, you configure the matplotlib backend dynamically before importing `pyplot`:
> ```python
> if not os.environ.get('DISPLAY', ''):
>     matplotlib.use('Agg')
> import matplotlib.pyplot as plt
> ```
> Ruff expects all module imports to be at the very top of the file (E402). If you place `from banknote_classifier...` imports *after* this `if` block, Ruff will raise a lint error.
> 
> To resolve this, place all standard library and package imports (including `banknote_classifier`) at the very top of `main.py`, followed by the environment check and backend selection, and finally `import matplotlib.pyplot as plt` at the end of the import block.

---

### Task 3D: Static Type Checking with `mypy`

Python is a dynamically typed language, but adding **type annotations** helps catch bugs early and makes your code much easier to read.

1. Add Mypy configuration to `pyproject.toml`:
   ```toml
   [tool.mypy]
   ignore_missing_imports = true
   ```
   > **Note**: We use basic mode here (not `strict = true`) to focus on learning type annotations without getting overwhelmed by PyTorch's complex internal types.

2. Add type annotations to your key functions. For example:
   ```python
   # Before (no types)
   def set_seed(seed):
       random.seed(seed)

   # After (with types)
   def set_seed(seed: int) -> None:
       random.seed(seed)
   ```

   Focus on annotating:
   - Function parameters and return types
   - Class `__init__` methods
   - The `ModelRegistry` class methods

3. Run the type checker:
   ```bash
   uv run mypy src/ main.py
   ```

**✅ Checkpoint**: `uv run mypy src/ main.py` reports no errors.

> **Bonus Challenge** 🏆: If you want to push further, try enabling strict mode by adding `strict = true` to the `[tool.mypy]` config and fixing all the additional type errors it surfaces. This is the gold standard for production Python code.
