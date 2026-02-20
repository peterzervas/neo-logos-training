# Neo-Logos Training Project Structure

## Directory Layout

```
neo-logos-training/
├── pyproject.toml
├── README.md
├── .gitignore
│
├── src/neo_logos/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py              # load_config() helper
│   │   ├── settings.py              # PROJECT_ROOT, DEFAULT_MODEL (claude-sonnet-4-6)
│   │   ├── cornerstone_arc.json     # 29-entry pre-planned memory narrative
│   │   ├── identity_categories.json # 7 identity categories with targets
│   │   ├── identity_parameters.json # Neo-Logos identity (Aetheron, Peter/Shaun/Jacob)
│   │   ├── identity_prompts.json    # Narrative-to-dialogue conversion prompts
│   │   └── eval_prompts.json        # Model evaluation prompts
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── env_loader.py            # .env file loading (explicit call required)
│   │   └── logging_utils.py         # Unified get_logger()
│   │
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base_generator.py        # Base class: API client, batch mode, caching, structured outputs
│   │   ├── identity_generator.py    # Identity narratives (extends BaseGenerator)
│   │   ├── enhanced_identity_generator.py  # Multi-format narratives via FormatManager
│   │   ├── articles_generator.py    # Neo-Ethics Q&A (lived experience voice)
│   │   ├── conversation_generator.py  # Multi-turn conversations (19 types)
│   │   ├── negative_examples_generator.py  # DPO chosen/rejected pairs (11 categories)
│   │   └── formats/
│   │       ├── __init__.py
│   │       ├── format_base.py       # NarrativeFormatGenerator base class
│   │       ├── format_manager.py    # Category-to-format routing
│   │       ├── cornerstone_generator.py
│   │       ├── reverie_generator.py
│   │       ├── bicameral_generator.py
│   │       ├── memory_generator.py
│   │       ├── self_generator.py
│   │       ├── narrative_generator.py
│   │       └── emotions_generator.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model_presets.py         # 3B/8B/27B/30B/70B hardware presets
│   │   ├── prepare_diverse_training.py  # Combine all data, 80/10/10 split, manifest
│   │   └── train_neo_logos.py       # Fine-tune Gemma 3 27B via Unsloth
│   │
│   └── scripts/
│       ├── generate_all.py          # Orchestrate all 4 generators in parallel
│       ├── run_full_pipeline.sh
│       ├── test_format_enhancements.sh
│       ├── convert_to_gguf.py
│       └── run_model_evaluation.py
│
├── corpus/
│   └── neo_ethics_articles/         # 16 articles (Articles 0-15)
│
├── dataset_outputs/                 # Generated training data (gitignored)
│   ├── neo_logos_identity/
│   ├── neo_logos_articles/
│   ├── conversations/
│   ├── dpo_pairs/
│   └── prepared/                    # Combined + split data with manifest
│
├── docs/
│   ├── NARRATIVE_FORMATS_STATUS.md
│   ├── NEO_LOGOS_FORMAT_CAPABILITIES.md
│   └── file-structure-doc.md
│
└── tests/
```

## Class Hierarchy

```
BaseGenerator (base_generator.py)
├── NeoIdentityGenerator (identity_generator.py)
│   └── EnhancedNeoIdentityGenerator (enhanced_identity_generator.py)
├── NeoArticlesGenerator (articles_generator.py)
├── ConversationGenerator (conversation_generator.py)
└── NegativeExamplesGenerator (negative_examples_generator.py)

NarrativeFormatGenerator (format_base.py)
├── CornerstoneMemoryGenerator
├── ReverieGenerator
├── BicameralMindGenerator
├── MemoryContinuityGenerator
├── SelfDialogueGenerator
├── NarrativeReflectionGenerator
└── EmotionsGenerator
```

## Training Target

**Gemma 3 27B** (`unsloth/gemma-3-27b-it`) - dense 27B parameter conversational model, fine-tuned with QLoRA (~22GB VRAM on RTX 5090).
