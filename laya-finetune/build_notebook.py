"""Write finetune_laya.ipynb: short cells in the order the story is told, each calling the
modules that produced the numbers."""

import nbformat as nbf

SOURCES = ["data/train_generated.jsonl", "data/hybrid_haiku.jsonl", "data/hybrid.jsonl"]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Fine-tuning Laya on a grading job, on Strands\n"
       "A small open decision model, specialised on one grading job, against a general hosted one.\n"
       "Every number below is computed in this run, except training itself, which takes about 85 minutes.\n\n"
       "The data, training, scoring and Jev cells are the original notebook. What moves to "
       "[Strands Agents](https://strandsagents.com) is Claude: every row of training data was written by Claude, "
       "and those calls now run on Amazon Bedrock as Strands agents with typed replies (`claude_bedrock.py`), "
       "kept under a dollar cap by a Strands hook."),
    code("%matplotlib inline\nfrom nb_helpers import *\nversions()"),

    md("## The data, built on first run\n"
       "Nothing in `data/` is shipped. `make_data.py` builds the 500 held-out items (Haiku writes, Sonnet 5 grades), "
       "the training rows (Haiku writes, planted labels only), the code-made surface defects, and `data/train_all.jsonl`, "
       "all on Bedrock through Strands. About $8 and 20 minutes the first time; every step skips what it already holds, "
       "so after that this cell only confirms the files. If you have run `laya-vs-jev-judge`, "
       "`make_data(test_from='../laya-vs-jev-judge/data/judged.jsonl')` reuses its test set."),
    code("from make_data import main as make_data\nmake_data()"),

    md("## The job, and the answer key\n"
       "Each item is a technical question and an AI-written answer with one **planted** defect, or none "
       "(one in three is clean). The planted label is the answer key both engines are scored against."),
    code("test = load_test()\ntest_summary(test)\nshow_item(test[7])"),

    md("## Step 0. Where the rows come from: Claude on Bedrock, through Strands\n"
       "`build_train_set.py` and `build_hybrid.py` ask Haiku for each item with `ask()`: one fresh Strands agent, "
       "one typed `QA` reply, no JSON to parse. The dollar cap is not in that code at all. It is `SpendCap`, a hook "
       "on every agent: before an invocation it cancels the call once the cap is reached, after it it adds the cost."),
    code("import inspect\nfrom claude_bedrock import GENERATOR_MODEL, HAIKU_USD, QA, SpendCap, ask\n"
         "print(inspect.getsource(SpendCap))"),
    md("Here it is live: ten items asked of Haiku one after another under a one-cent cap. "
       "The cap stops the run part-way, and the new items go through the same test-copy check as Step 2."),
    code("import random\nfrom build_train_set import TOPICS, gen_prompt\nfrom prepare_data import drop_test_copies\n\n"
         "cap, rng, fresh = SpendCap(0.01, *HAIKU_USD), random.Random(7), []  # nosec B311: seeded demo data\n"
         "for i, planted in enumerate(['none', 'fabricated_number', 'unit_error', 'none', 'evasive',\n"
         "                             'incomplete', 'none', 'wrong_entity', 'truncated', 'none']):\n"
         "    topic = rng.choice(TOPICS)\n"
         "    pair, usage = ask(GENERATOR_MODEL, gen_prompt(topic, planted), QA, hooks=[cap])\n"
         "    if pair:\n"
         "        fresh.append({'id': 900_000 + i, 'planted': planted, 'topic': topic, **pair.model_dump()})\n"
         "print(f'{GENERATOR_MODEL}: {len(fresh)} items written, ${cap.spent:.4f} spent, '\n"
         "      f'{cap.cancelled} calls cancelled by the cap before reaching Claude')\n"
         "show_item(fresh[1])\n"
         "_, near = drop_test_copies(fresh)\n"
         "print(f'near-copies of a test question: {len(near)} of {len(fresh)}')"),

    md("## Step 1. Combine everything that was generated\n"
       "Haiku wrote every question and answer. The code-made rows plant surface defects with templates."),
    code(f"from prepare_data import combine, drop_test_copies, weighted\nSOURCES = {SOURCES!r}\n"
         "sources_summary(SOURCES)\nraw = combine(SOURCES)\nprint(len(raw), 'rows after removing duplicate ids')"),

    md("## Step 2. Drop every near-copy of a test question\n"
       "Haiku repeats itself inside a topic. Training on a reworded test question teaches the answer, not the task."),
    code("kept, dropped = drop_test_copies(raw)\nprint(f'dropped {len(dropped)} of {len(raw)}, kept {len(kept)}')\n"
         "show_near_copy(dropped)"),

    md("## Step 3. Look at the class mix before training\n"
       "**Lesson 1:** a clean-up step can break the mix. Clean questions are generic, so most near-copies were "
       "clean answers, and dropping them pushed the clean share well below the one in three the data was built with."),
    code("from mix_report import plot, print_table, problems, shares\n"
         "stages = [('as generated', raw, False), ('after dropping test copies', kept, False)]\n"
         "print_table(stages)\nplot(stages, 'plots/mix_step3.png');"),

    md("## Step 4. Weight back to the design, and check every defect type\n"
       "Weighting the kept rows back to one clean in three fixes the clean share. Check the line under the table: "
       "if some defect types are still off their equal share, only `--mix balanced` puts this data on design, and "
       "that is what Step 5 trains with. The target comes from how the data was generated, never from the "
       "test set's labels."),
    code("design = weighted(kept, 'design')\n"
         "stages = [('as generated', raw, False), ('after dropping test copies', kept, False),\n"
         "          ('weighted: clean 1 in 3', design, True)]\n"
         "print_table(stages)\nprint('; '.join(problems(shares(design, True))) or 'on design')\n"
         "plot(stages, 'plots/mix_fixed.png', compare=(1, 2));"),
    md("**Lesson 2:** check again after every change. Adding 40 natural examples to 11 defect types doubled "
       "their share; a model trained on that mix over-called those 11 types. Weighting clean vs defective "
       "does not catch it. Balancing every type does."),
    code("natural, _ = drop_test_copies(combine(['data/natural_haiku.jsonl']))\nmore = kept + natural\n"
         "stages = [('+ natural, unweighted', more, False),\n"
         "          ('+ natural, weighted: clean 1 in 3', weighted(more, 'design'), True),\n"
         "          ('+ natural, types balanced too', weighted(more, 'balanced'), True)]\n"
         "print_table(stages)\n"
         "for name, rows, w in stages:\n    print(f'{name:36s}', '; '.join(problems(shares(rows, w))) or 'on design')\n"
         "plot(stages, 'plots/mix_step4.png', compare=(1, 2));"),

    md("## Step 5. Train\n"
       "`train.py` repeats this check before its first step and refuses a skewed mix. The full run takes about "
       "85 minutes on an M4 Pro; here are its first 25 steps, so you can watch it start. "
       "The checkpoint scored below is the finished run."),
    code("import sys  # the kernel's own Python, so the shell command runs in this venv\n"
         "!{sys.executable} train.py --train data/train_all.jsonl --mix balanced --max-steps 25 2>&1 | grep -v -i warn"),

    md("## Step 6. Score on the 500 held-out items\nLaya out of the box, then fine-tuned, on the same items and questions."),
    code("from evaluate import evaluate\nzero_shot = evaluate('base')"),
    code("tuned = evaluate('runs/all2082')"),

    md("## Jev on the same 500 items, same short labels\nA hosted API, so its latency includes the network."),
    code("jev = await run_jev(test)"),

    md("## The scoreboard"),
    code("scoreboard({'Laya, out of the box': zero_shot, 'Laya, fine-tuned': tuned, 'Jev': jev})"),
    code("print('pass/rework, agreement on the most confident share of items:')\n"
         "for share, acc in tuned['verdict']['risk_coverage']:\n    print(f'  top {share:>4.0%}: {acc:.3f}')"),

    md("## Before you train on your own data\n"
       "1. No test copies: drop every training item that is a near-copy of a test item.\n"
       "2. Check the mix after **every** change to the data, clean vs defective and type by type, "
       "against how production looks, not against your test labels.\n"
       "3. If you planted the labels, you already have them: a paid judge is a quality check, not a requirement."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "finetune_laya.ipynb")
print("wrote finetune_laya.ipynb with", len(cells), "cells")
