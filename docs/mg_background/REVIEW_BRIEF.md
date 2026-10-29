# Review brief: MDWF multigrid note (fresh-session read-through)

You have not seen how this paper was written. Read it as a referee would and list problems; **do not edit
any file**. Report in Traditional Chinese, one item per problem: page/section, quoted sentence, what is
wrong, the evidence (file + row or log line).

## Files
- Paper: `MDWF_MULTIGRID_NOTE.tex` (main, appendices) + `results.tex` (Sec. V). Built PDF: `MDWF_MULTIGRID_NOTE.pdf`.
- Provenance: `REPRODUCE.md` (table → commit → program → data file). Data: `data/*.tsv`.
- Run directories (logs) are under `/home/chen/red_gpu/ruttice_a30_20261016/` (`multiconf/p1`, `multiconf/p2`,
  `massscan_newconf`, `multiconf/herm`, `gcr_newconf`); plan and classification in `multiconf/CLASSIFICATION.md`.

## What changed on 2026-10-28/29 (check these hardest)
Most tables moved from one configuration to four per volume (configurations 1–4, Table "configurations").
Means, medians or ranges are stated in each caption. Check that:
1. every number in the text matches its table, and every table cell matches its data file or log;
2. no sentence still describes one configuration where the table now has four (abstract, Sec. V.A,
   V.B, V.C, V.D, conclusions, Open problems); no internal names (v2, 702, 802, 902, seed lists) in the text;
3. comparison words are not stronger than the data ("never the fastest" ≠ "slower in every case");
4. each conclusion appears once per abstract / intro / results / conclusions, not repeated between sections;
5. table and figure numbers in `REPRODUCE.md` match the PDF;
6. claims marked as inference are not stated as fact elsewhere.

## Changed at the author's request on 10-29 (check the wording)
- Sec. V.B.2: the D† stall is now attributed to the small smeared lattice, not to Q.
- Table "sensitivity", setup B, 8 test vectors: rerun with 16 filtered vectors.

## Style rules (from the author)
Run `python3 -I ~/.claude/tools/style_lint.py <file>` on both .tex files. American spelling. No em-dash,
no "not X but Y" / "X, not Y", no dramatic short openers, no self-narration ("this section shows"),
no process history ("a previous version"), "per" not "a", first sentence of a paragraph states the result,
symbols defined before use, numbers with units and conditions, "ILU" not "DILU" (n/a here).
