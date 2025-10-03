# Data Distillation & Persona-based Knowledge Rewriting

This repository demonstrates methods for **distilling knowledgeable data** using **Virtual Indian Personas**.
The approach leverages personas with diverse cultural and professional backgrounds to **extract, expand, and rewrite domain-specific knowledge** in a more contextual, human-like way.

---

## Methods for Persona Synthesis

### 1. Text-to-Persona

* Any piece of text reflects the type of reader or writer who might produce or engage with it.
* By prompting an LLM with such texts, we infer and generate personas that are likely to **read, write, like, or dislike** that content.
* Since web text is broad and virtually unlimited, this method yields **wide-ranging persona coverage** across multiple domains.

### 2. Persona-to-Persona

* Text-to-Persona may miss personas with **low online visibility** (e.g., children, beggars, backstage crew).
* Persona-to-Persona addresses this by generating new personas from **interpersonal relationships** with existing ones.
* Example: A “film director” persona could generate related personas like “lighting technician” or “background dancer.”

---

## Knowledge Distillation Workflow

Each Virtual Indian Persona has a **specific knowledge domain**. We use this structure to **ask thought-provoking questions** and obtain distilled, domain-rich knowledge.

**Steps:**

1. **Persona Population** – Synthesize diverse personas across knowledge domains.
2. **Question Generation** – Create domain-specific, thought-provoking prompts.
3. **Knowledge Extraction** – Query virtual personas to obtain rewritten/distilled knowledge.

> 💡 Personas can also be grouped by domain to perform **major knowledge distillation** (e.g., multiple personas in *biomedical ethics*).

Example prompt templates are available in **`dd_instruction.yml`**.

---

## Setup

* Use [SGLang](https://github.com/sglang) or [vLLM](https://github.com/vllm-project/vllm) to load the model locally.
* Run inference with asynchronous requests:

```bash
python async_infer.py --model <your_model_path> --input <input_file> --output <output_file>
```

---

## Example: Persona → Question → Distilled Knowledge

**Persona**

> A scientist who advocates for the ethical use of stem cells in research, particularly embryonic stem cells.
> Domain: *Stem Cell Biology & Ethics*

**Thought-Provoking Question**

> How should India navigate the ethical complexities of stem cell research and therapy, balancing the potential for groundbreaking medical advancements with moral considerations like embryonic stem cells, informed consent, and equitable access?

**Generated Knowledge**

* Detailed exploration of **ethical dilemmas**, **cultural values**, and **regulatory frameworks** in India.
* Emphasis on **informed consent**, **public healthcare access**, and **alternative research methods** (like iPSCs).
* Provides a **vision of India as a global leader** in ethical biomedical innovation.

(See full example in the `opensource-release/training-corpus/indian-english ` directory.)

---

## Applications

* Creating **knowledge-rich synthetic datasets** for LLM training.
* Domain-specific **question answering** and **policy exploration**.
* Cultural and ethical grounding of scientific or social debates.

---
