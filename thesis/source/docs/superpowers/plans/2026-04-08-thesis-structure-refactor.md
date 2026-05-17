# Thesis Structure Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic `njuthesis-sample.tex` workflow with a formal `thesis.tex` entry file plus separated front matter, chapter, and appendix source files that are easier to maintain for a 100-page thesis.

**Architecture:** Keep the existing `njuthesis` template and setup files intact while moving document content into `frontmatter/`, `chapters/`, and `appendices/`. Use `\input` for short front matter units and `\include` for chapter-scale content so the document stays easy to compile and later supports `\includeonly`.

**Tech Stack:** LaTeX (`njuthesis`), XeLaTeX, Biber, Git

---

## File Structure

- Modify: `thesis.tex` (renamed from `njuthesis-sample.tex`)
- Create: `frontmatter/abstract-zh.tex`
- Create: `frontmatter/abstract-en.tex`
- Create: `frontmatter/acknowledgement.tex`
- Create: `chapters/01-introduction.tex`
- Create: `chapters/02-related-work.tex`
- Create: `chapters/03-method.tex`
- Create: `chapters/04-experiments.tex`
- Create: `chapters/05-conclusion.tex`
- Create: `appendices/appendix-a.tex`
- Delete: `njuthesis-sample.tex`

## Verification Strategy

This refactor operates on LaTeX document/configuration files, so classic unit-test-first TDD does not apply cleanly. Verification will use document builds after each structural task:

- `xelatex -interaction=nonstopmode -halt-on-error thesis.tex`
- `biber thesis`
- `xelatex -interaction=nonstopmode -halt-on-error thesis.tex`
- `xelatex -interaction=nonstopmode -halt-on-error thesis.tex`

Expected final result:
- exit code `0` for each command
- `thesis.pdf` generated
- no `LaTeX Error` or `Undefined control sequence` in `thesis.log`

### Task 1: Establish the new main entry file and front matter split

**Files:**
- Create: `frontmatter/abstract-zh.tex`
- Create: `frontmatter/abstract-en.tex`
- Create: `frontmatter/acknowledgement.tex`
- Modify: `thesis.tex`
- Delete: `njuthesis-sample.tex`

- [ ] **Step 1: Extract the current Chinese abstract into `frontmatter/abstract-zh.tex`**

- [ ] **Step 2: Extract the current English abstract into `frontmatter/abstract-en.tex`**

- [ ] **Step 3: Extract the current acknowledgement into `frontmatter/acknowledgement.tex`**

- [ ] **Step 4: Rename `njuthesis-sample.tex` to `thesis.tex` and replace inline front matter blocks with `\input{...}` statements**

- [ ] **Step 5: Add a brief comment in `thesis.tex` explaining that it is the orchestration file and daily writing should happen in the chapter files**

- [ ] **Step 6: Run an initial syntax/build check**

Run:
```powershell
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
```

Expected:
- command exits `0`
- `thesis.aux` and `thesis.pdf` are generated

- [ ] **Step 7: Commit**

```bash
git add thesis.tex frontmatter/abstract-zh.tex frontmatter/abstract-en.tex frontmatter/acknowledgement.tex
git rm njuthesis-sample.tex
git commit -m "refactor: split thesis front matter"
```

### Task 2: Split body chapters and appendices into dedicated files

**Files:**
- Modify: `thesis.tex`
- Create: `chapters/01-introduction.tex`
- Create: `chapters/02-related-work.tex`
- Create: `chapters/03-method.tex`
- Create: `chapters/04-experiments.tex`
- Create: `chapters/05-conclusion.tex`
- Create: `appendices/appendix-a.tex`

- [ ] **Step 1: Move the existing sample body content into `chapters/01-introduction.tex` with its own `\chapter{...}` declaration**

- [ ] **Step 2: Create placeholder chapter files for `02-related-work`, `03-method`, `04-experiments`, and `05-conclusion`**

- [ ] **Step 3: Move the current appendix sample content into `appendices/appendix-a.tex` with its own `\chapter{...}` declaration**

- [ ] **Step 4: Replace inline body and appendix content in `thesis.tex` with `\include{...}` statements**

- [ ] **Step 5: Run the full bibliography-aware build**

Run:
```powershell
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
biber thesis
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
```

Expected:
- all commands exit `0`
- `thesis.pdf` is updated
- bibliography and table of contents resolve

- [ ] **Step 6: Check the log for structural errors**

Run:
```powershell
Select-String -Path 'thesis.log' -Pattern 'LaTeX Error|Undefined control sequence|Emergency stop|Fatal error'
```

Expected:
- no matches

- [ ] **Step 7: Commit**

```bash
git add thesis.tex chapters appendices
git commit -m "refactor: split thesis chapters and appendices"
```

### Task 3: Final verification and cleanup

**Files:**
- Review: `thesis.tex`
- Review: `frontmatter/*.tex`
- Review: `chapters/*.tex`
- Review: `appendices/*.tex`

- [ ] **Step 1: Confirm the new tree shape**

Run:
```powershell
Get-ChildItem frontmatter,chapters,appendices | Format-Table -AutoSize Name,Length,LastWriteTime
```

Expected:
- all planned files exist

- [ ] **Step 2: Confirm Git status only contains intended structural changes**

Run:
```powershell
git status --short
```

Expected:
- only the new thesis structure changes appear before the final commit
- clean working tree after the final commit

- [ ] **Step 3: Record the final implementation commit**

```bash
git log --oneline -n 3
```

Expected:
- shows the front matter and chapter-splitting commits on `feature/thesis-structure-refactor`
